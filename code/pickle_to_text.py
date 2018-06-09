import pickle
import sys

file_in = sys.argv[1]
file_out = sys.argv[2]

with open(file_in, 'rb') as fp:
    stats_dict = pickle.load(fp)

with open(file_out, 'w') as fp:
    fp.write('Training loss\n')
    for i in range(len(stats_dict['loss'])):
        fp.write(str(stats_dict['loss'][i]))
        fp.write('\n')
    fp.write('\n')
    fp.write('Training accuracy\n')
    for i in range(len(stats_dict['loss'])):
        fp.write(str(stats_dict['categorical_accuracy'][i]))
        fp.write('\n')
    fp.write('\n')
    fp.write('Validation loss\n')
    for i in range(len(stats_dict['loss'])):
        fp.write(str(stats_dict['val_loss'][i]))
        fp.write('\n')
    fp.write('\n')
    fp.write('Validation accuracy\n')
    for i in range(len(stats_dict['loss'])):
        fp.write(str(stats_dict['val_categorical_accuracy'][i]))
        fp.write('\n')
