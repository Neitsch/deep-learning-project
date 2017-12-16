from subprocess import call

if __name__ == "__main__":
    mean = 0.05
    while mean < 0.41:
        mean_str = str(mean)[0:4].replace('.','_')
        command = 'python train.py -g 0 --save saved_model_{} --reconstruct --noise={} --epoch=15'.format(mean_str, mean)
        print("running command: {}".format(command))
        call(command)
        mean += 0.05
