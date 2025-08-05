ENT="python vqvae.py "
cfg="config/vqvae.yaml"
ckpts=""
saveroot=""
exp_name=""
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --ckpts $ckpts \
    --save_root $saveroot \
    --exp_name $exp_name \
    --test
