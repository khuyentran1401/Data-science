import sys

model_type = sys.argv[1]
model_version = sys.argv[2]
model_path = f'''model/model1/{model_type}/version_{model_version}'''
print('Loading model from', model_path, 'for training')

# On the terminal type
# for version in 1 2 3 4 
# do 
# python train.py $version 
# done