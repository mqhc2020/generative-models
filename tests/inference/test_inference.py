import numpy
from PIL import Image
import pytest
from pytest import fixture
import torch
from typing import Tuple

from sgm.inference.api import (
    model_specs,
    SamplingParams,
    SamplingPipeline,
    Sampler,
    Sampler_mini,
    ModelArchitecture,
)
import sgm.inference.helpers as helpers

import os
import numpy as np

# TODO: besides batch size control, add iteration control also?
NUM_SAMPLES = 1

N_STEPS = 40
HIGH_NOISE_FRAC = 0.8


@pytest.mark.inference
class TestInference:
    @fixture(scope="class", params=model_specs.keys())
    def pipeline(self, request) -> SamplingPipeline:
        pipeline = SamplingPipeline(request.param)
        yield pipeline
        del pipeline
        torch.cuda.empty_cache()

    @fixture(
        scope="class",
        params=[
            [ModelArchitecture.SDXL_V1_BASE, ModelArchitecture.SDXL_V1_REFINER],
#            [ModelArchitecture.SDXL_V0_9_BASE, ModelArchitecture.SDXL_V0_9_REFINER],
        ],
        ids=["SDXL_V1"],
#        ids=["SDXL_V1", "SDXL_V0_9"],
    )
    def sdxl_pipelines(self, request) -> Tuple[SamplingPipeline, SamplingPipeline]:
        base_pipeline = SamplingPipeline(request.param[0])
        refiner_pipeline = SamplingPipeline(request.param[1])
        yield base_pipeline, refiner_pipeline
        del base_pipeline
        del refiner_pipeline
        torch.cuda.empty_cache()

    def create_init_image(self, h, w):
        image_array = numpy.random.rand(h, w, 3) * 255
        image = Image.fromarray(image_array.astype("uint8")).convert("RGB")
        return helpers.get_input_image_tensor(image)

    '''
    @pytest.mark.parametrize("sampler_enum", Sampler)
    def test_txt2img(self, pipeline: SamplingPipeline, sampler_enum):
        output = pipeline.text_to_image(
            params=SamplingParams(sampler=sampler_enum.value, steps=N_STEPS),
            prompt="A professional photograph of an astronaut riding a pig",
            negative_prompt="",
            samples=1,
        )

        assert output is not None

        NUM_SAMPLES = 1

        output = output.cpu() * 255
        for i in range(0, NUM_SAMPLES):
            # N C H W
            _sample = output[i]
            _sample = np.transpose(_sample.numpy().astype(np.uint8), (1, 2, 0))
            print(f"Shape of tensor (rearranged): {_sample.shape}")
            img = Image.fromarray(_sample)
            img.save(os.path.join('output/txt2img', 'output_{}_{}.png'.format(sampler_enum.value, i)))


    @pytest.mark.parametrize("sampler_enum", Sampler)
    def test_img2img(self, pipeline: SamplingPipeline, sampler_enum):
        output = pipeline.image_to_image(
            params=SamplingParams(sampler=sampler_enum.value, steps=N_STEPS),
            image=self.create_init_image(pipeline.specs.height, pipeline.specs.width),
            prompt="A professional photograph of an astronaut riding a pig",
            negative_prompt="",
            samples=1,
        )
        assert output is not None

        NUM_SAMPLES = 1

        output = output.cpu() * 255
        for i in range(0, NUM_SAMPLES):
            # N C H W
            _sample = output[i]
            _sample = np.transpose(_sample.numpy().astype(np.uint8), (1, 2, 0))
            print(f"Shape of tensor (rearranged): {_sample.shape}")
            img = Image.fromarray(_sample)
            img.save(os.path.join('output/img2img', 'output_{}_{}.png'.format(sampler_enum.value, i)))
    '''

    @pytest.mark.parametrize("sampler_enum", Sampler_mini)
    @pytest.mark.parametrize(
        "use_init_image", [False], ids=["txt2img"]
        #"use_init_image", [True, False], ids=["img2img", "txt2img"]
    )
    def test_sdxl_with_refiner(
        self,
        sdxl_pipelines: Tuple[SamplingPipeline, SamplingPipeline],
        sampler_enum,
        use_init_image,
        n_samples
    ):
        base_pipeline, refiner_pipeline = sdxl_pipelines

        #ret = pytestconfig.getoption('n_samples')
        #ret = n_samples
        #samples = int(ret) if ret != None else NUM_SAMPLES
        print('n_samples =', n_samples)

        if use_init_image:
            output = base_pipeline.image_to_image(
                params=SamplingParams(sampler=sampler_enum.value, steps=int(N_STEPS*HIGH_NOISE_FRAC)),
                image=self.create_init_image(
                    base_pipeline.specs.height, base_pipeline.specs.width
                ),
                prompt="A professional photograph of an astronaut riding a pig",
                negative_prompt="",
                samples=n_samples,
                return_latents=True,
            )
        else:
            output = base_pipeline.text_to_image(
                params=SamplingParams(sampler=sampler_enum.value, steps=int(N_STEPS*HIGH_NOISE_FRAC)),
                prompt="A professional photograph of an astronaut riding a pig",
                negative_prompt="",
                samples=n_samples,
                return_latents=True,
            )

        assert isinstance(output, (tuple, list))
        samples, samples_z = output
        assert samples is not None
        assert samples_z is not None
        output = refiner_pipeline.refiner(
            params=SamplingParams(sampler=sampler_enum.value, steps=int(N_STEPS*(1-HIGH_NOISE_FRAC))),
            image=samples_z,
            prompt="A professional photograph of an astronaut riding a pig",
            negative_prompt="",
            samples=n_samples,
        )

        output = output.cpu() * 255
        for i in range(0, n_samples):
            # N C H W
            _sample = output[i]
            _sample = np.transpose(_sample.numpy().astype(np.uint8), (1, 2, 0))
            #print(f"Shape of tensor (rearranged): {_sample.shape}")
            img = Image.fromarray(_sample)
            img.save(os.path.join('profiling/outputs', 'output_{}_{}_{}_b{}.png'.format(sampler_enum.value, 'ii' if use_init_image else 'ti', i, n_samples)))

