#!/bin/bash
# Working directory is still under StableDiffusion
# Todo: put the command as argument, just like merge.sh ("python scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt 512-base-ema.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda")

PROF_DIR="profiling"
TUNING_DIR="tuning_results"
LOG_DIR="logs"
ROCBLASTER_OUTPUT="./BlasterOutput.csv"
ROCTUNER_OUTPUT="./TunerOutput.csv"
ROC_SOLUTIONS_FILE_PREFIX="./Solutions"
TUNED_CSV_RESULT=$PROF_DIR/blas_tuning_applied.csv
NOT_TUNED_CSV_RESULT=$PROF_DIR/profiling/blas_tuning_not_applied.csv
TUNED_STATS_OUTPUT=$PROF_DIR/blas_tuning_applied.stats.csv
SAMPLING_CSV_FILE_IN="sampling.csv"
SAMPLING_CSV_FILE_OUT="sampling_out.csv"

FAST_MODE=False
NUM_ITER=5
BATCH_SIZE_UP_TO=8


function wait_for_subs() {
	for job in `jobs -p`
	do
	echo $job
	    wait $job || let "FAIL+=1"
	done
}

function get_opt() {
	if [[ "$SD_PT_MEM" == "" ]]; then
                ATTN_BACKEND="no_opt"
        else
                ATTN_BACKEND=$SD_PT_MEM
        fi

	echo $ATTN_BACKEND
}

# http://phodd.net/gnu-bc/bcfaq.html#bashlog
function log()
{
        local x=$1
        n=2
        l=-1
        if [ "$2" != "" ];
        then
                n=$x
                x=$2
        fi
        while((x));
        do
                let l+=1 x/=n
        done;
        echo $l;
}

if [[ "$1" == "-X" || "$1" == "-x" ]]; then

	if [[ "$1" == "-X" ]]; then
		FAST_MODE=True
		#NUM_ITER=1
		#BATCH_SIZE_UP_TO=1
	fi

	command -v rocBlaster >/dev/null 2>&1 || { echo >&2 "Need rocBlaster but it's not installed. Aborting."; exit 1; }

	mkdir -p $PROF_DIR/$TUNING_DIR
	ATTN_BACKEND=$(get_opt)

	if [[ "$1" == "-X" ]]; then
		for b in $(seq 1 $BATCH_SIZE_UP_TO)
		do
                        OUTPUT_FILE=$PROF_DIR/$TUNING_DIR/${ROC_SOLUTIONS_FILE_PREFIX}_${ATTN_BACKEND}_$b.csv
			HIP_VISIBLE_DEVICES=$(($b-1)) rocBlaster -o $OUTPUT_FILE python scripts/txt2img.py --n_samples $b --prompt "a professional photograph of an astronaut riding a horse" --ckpt 512-base-ema.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda &
		done
		wait_for_subs
	else
		for b in $(seq 1 $BATCH_SIZE_UP_TO)
		do
			OUTPUT_FILE=$PROF_DIR/$TUNING_DIR/${ROC_SOLUTIONS_FILE_PREFIX}_${ATTN_BACKEND}_$b.csv
			rocBlaster -o $OUTPUT_FILE python scripts/txt2img.py --n_samples $b --prompt "a professional photograph of an astronaut riding a horse" --ckpt 512-base-ema.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda
		done
	fi
	echo "$OUTPUT_FILE written."

elif [[ "$1" == "-y" ]]; then

	mkdir -p $PROF_DIR/$TUNING_DIR
        ATTN_BACKEND=$(get_opt)

	for b in $(seq 1 $BATCH_SIZE_UP_TO)
        do
            OUTPUT_FILE=$PROF_DIR/$TUNING_DIR/${ROC_SOLUTIONS_FILE_PREFIX}_${ATTN_BACKEND}_$b.csv
            python $PROF_DIR/parse_stats_rocm.py -t --csv-file $OUTPUT_FILE python scripts/txt2img.py --n_samples $b --prompt "a professional photograph of an astronaut riding a horse" --ckpt 512-base-ema.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda
	done


elif [[ "$1" == "-r" ]]; then

	rocprof --stats -o $NOT_TUNED_CSV_RESULT python scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt 512-base-ema.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda
	ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=$ROCBLASTER_OUTPUT rocprof --stats -o $TUNED_CSV_RESULT python scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt 512-base-ema.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda

elif [[ "$1" == "-T" || "$1" == "-t" ]]; then

	if [[ -d $PROF_DIR/$TUNING_DIR ]]; then

		mkdir -p $PROF_DIR/$LOG_DIR

		if [[ "$1" == "-T" ]]; then
			FAST_MODE=True
			#NUM_ITER=1
			#BATCH_SIZE_UP_TO=1
		fi

		for i in $(seq 1 $NUM_ITER)
		do
			if [[ "$FAST_MODE" == "True" ]]; then
				for b in $(seq 1 $BATCH_SIZE_UP_TO)
				do
					HIP_VISIBLE_DEVICES=$(($b-1)) python scripts/txt2img.py --n_samples $b --prompt "a professional photograph of an astronaut riding a horse" --ckpt 512-base-ema.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda &
				done

				wait_for_subs
				echo "<iter $i> <batch $BATCH_SIZE_UP_TO> finished. Not tuned."

				ATTN_BACKEND=$(get_opt)
				INPUT_FILE=$PROF_DIR/$TUNING_DIR/${ROC_SOLUTIONS_FILE_PREFIX}_${ATTN_BACKEND}_$b.csv
				for b in $(seq 1 $BATCH_SIZE_UP_TO)
				do
					HIP_VISIBLE_DEVICES=$(($b-1)) ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=$INPUT_FILE python scripts/txt2img.py --n_samples $b --prompt "a professional photograph of an astronaut riding a horse" --ckpt 512-base-ema.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda
				done

				wait_for_subs
				echo "<iter $i> <batch $BATCH_SIZE_UP_TO> finished. Tuned."

			else
				#THE_TIME=$(date +"%Y-%m-%d")
				THE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
				LOG_FILE=$PROF_DIR/$LOG_DIR/sd_log_${THE_TIME}.log
				for b in $(seq 1 $BATCH_SIZE_UP_TO)
				do
					#python scripts/txt2img.py --n_samples $b --prompt "a professional photograph of an astronaut riding a horse" --ckpt 512-base-ema.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda >> $LOG_FILE
					pytest --n_samples $b -s tests/inference/test_inference.py

					set -x
					ATTN_BACKEND=$(get_opt)
					INPUT_FILE=$PROF_DIR/$TUNING_DIR/${ROC_SOLUTIONS_FILE_PREFIX}_${ATTN_BACKEND}_$b.csv
					ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=$INPUT_FILE python scripts/txt2img.py --n_samples $b --prompt "a professional photograph of an astronaut riding a horse" --ckpt 512-base-ema.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda >> $LOG_FILE
					echo "$INPUT_FILE read."
					set +x
				done
			fi
		done

	else
		echo "Can't find $PROF_DIR/$TUNING_DIR"
	fi
elif [[ "$1" == "-s" ]]; then

	expo=$(log $2)

	THE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
	LOG_FILE=$PROF_DIR/$LOG_DIR/sd_log_${THE_TIME}.log

	rm  $PROF_DIR/$SAMPLING_CSV_FILE_OUT

	for i in $(seq 0 $expo)
	do
		set -x
		b=$((2**$i))
		pytest --n_samples $b -s -v --log-file=$LOG_FILE tests/inference/test_inference.py
		python $PROF_DIR/utils.py -m -i $PROF_DIR/$SAMPLING_CSV_FILE_IN -o $PROF_DIR/$SAMPLING_CSV_FILE_OUT
		set +x
	done
elif [[ "$1" == "-p" ]]; then
	# python profiling/parse_stats_rocm.py -t --blas-clients-location=/dockerx/rocBLAS/build/release/clients/staging/ --prof-stats-file profiling/blas_rebuilt.stats.csv --util 0.8

	python $PROF_DIR/parse_stats_rocm.py -t --blas-clients-location=/dockerx/rocBLAS/build/release/clients/staging/ --prof-stats-file $TUNED_STATS_OUTPUT --util 0.8
	

elif [[ "$1" == "-h" ]]; then

	echo "-x : Tune for all batch sizes by rocBLASTER."
	echo "-X : Tune for all batch sizes by rocBLASTER. Fast mode."
	echo "-y : Tune for all batch sizes by parse_stats_rocm.py."
	echo "-r : Run rocprof for tuned and not-tuned cases."
	echo "-t : Compare running time between tuned and not-tuned cases."
	echo "-T : Compare running time between tuned and not-tuned cases. Fast mode."
	echo "-s : run with batch sizes 1, 2, 4, 8, ..."
	echo "-p : Parse rocprof stats and do estimation."
	echo "-h : help"

else

	echo "Wrong input."

fi
