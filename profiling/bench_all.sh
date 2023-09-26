mkdir -p output
pytest -v -s --log-file=./pytest.log tests/inference/test_inference.py
