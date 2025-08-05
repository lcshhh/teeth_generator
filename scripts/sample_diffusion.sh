ENT="python diffusion.py "
cfg="config/DiffusionVAE.yaml"
saveroot=""
exp_name=""
launcher="none"
ckpts=""



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --ckpt $ckpts \
    --sample

