ENT="python vqvae.py "
cfg="config/vqvae.yaml"
saveroot=""
exp_name=""
ckpts=""
launcher="none"


$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
