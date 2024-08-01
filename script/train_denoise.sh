python3 ./train/train_denoise.py --arch Uformer_B --batch_size 3 --gpu '0' \
   --train_ps 128 --train_dir txt/train.txt --env _0708 \
   --val_dir txt/test.txt --save_dir ./logs/ \
   --dataset car --warmup --pretrain_weights logs/denoising/car/Uformer_B_0708/models/model_best.pth --resume

# python3 ./train/train_denoise.py --arch Restormer --batch_size 1 --gpu '0' \
#     --train_ps 128 --train_dir txt/train.txt --env _0619 \
#     --val_dir txt/test.txt --save_dir ./logs/ \
#     --dataset car --warmup # --pretrain_weights logs/denoising/car/Uformer_B_0611/models/model_best.pth --resume


# python3 ./train/train_denoise.py --arch UNet --batch_size 2 --gpu '0' \
#     --train_ps 128 --train_dir txt/train.txt --env _0603 \
#     --val_dir txt/test.txt --save_dir ./logs/ \
#     --dataset car --warmup # --pretrain_weights logs/denoising/car/Uformer_B_0531/models/model_best.pth --resume
