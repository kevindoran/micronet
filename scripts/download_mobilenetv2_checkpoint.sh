wget 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz'
mkdir './resources/mobilenetv2_checkpoints/'
mkdir './resources/mobilenetv2_checkpoints/1.0_224/'
tar -xzf ./mobilenet_v2_1.0_224.tgz -C './resources/mobilenetv2_checkpoints/1.0_224/'
rm ./mobilenet_v2_1.0_224.tgz
