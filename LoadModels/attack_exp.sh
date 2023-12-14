# for i in {1..10}
# do
#     python RunAttacks.py >> adv_data_exp.txt
# done
CUDA_VISIBLE_DEVICES=1 python RunAttacks.py --model /mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/Vanilla/resnet164.pth \
--model1 /mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/Vanilla/resnet164.pth \
--model2 /mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/Vanilla/resnet164.pth \
> resnet164.txt