rm *.onnx
./run.sh --skip_data_prep false --skip_train true --download_model kamo-naoyuki/aishell_conformer --onnx_inference true --model_exporting true
