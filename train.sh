# model='bert-base-chinese'
# model='hfl/chinese-roberta-wwm-ext'
model='luhua/chinese_pretrain_mrc_macbert_large'
echo ${model}

echo 'Traning Multiple Choices'
python3.9 train_MC.py --model_name ${model}

echo "Training Question Answering"
python3.9 train_QA.py --model_name ${model}

echo "Predict Testing Data"
python3.9 test.py --model_name ${model}