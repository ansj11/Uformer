
# python3 test/test.py --input_dir txt/test.txt --result_dir logs/denoising/car/Uformer_B_0708/test/ --weights logs/denoising/car/Uformer_B_0708/models/model_best.pth


### test on SIDD ###
python3 test/test_denoise.py --input_dir ./outputs/jietuV9/demo_frames/ --result_dir ./outputs/jietuV9/denoise/ --weights logs/denoising/car/Uformer_B_0516/models/model_best.pth

### test on DND ###
# python3 test/test_dnd.py --input_dir ../datasets/denoising/dnd/input/ --result_dir ./results/denoising/DND/ --weights ./logs/denoising/SIDD/Uformer_B/models/model_best.pth 


### test on GoPro ###
# python3 test/test_gopro_hide.py --input_dir ../datasets/deblurring/GoPro/test/ --result_dir ./results/deblurring/GoPro/Uformer_B/ --weights ./logs/motiondeblur/GoPro/Uformer_B/models/model_best.pth

### test on HIDE ###
# python3 test/test_gopro_hide.py --input_dir ../datasets/deblurring/HIDE/test/ --result_dir ./results/deblurring/HIDE/Uformer_B/ --weights ./logs/motiondeblur/GoPro/Uformer_B/models/model_best.pth

### test on RealBlur ###
# python3 test/test_realblur.py --input_dir ../datasets/deblurring/ --result_dir ./results/deblurring/ --weights ./logs/motiondeblur/GoPro/Uformer_B/models/model_best.pth
