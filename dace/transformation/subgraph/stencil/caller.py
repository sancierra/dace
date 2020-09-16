from subprocess import call 
import sys 


filename = sys.argv[1]
with open('config.txt', 'r') as config:
    with open('log.txt','w') as log:
        for line in config:
            line = line.rstrip('\n')
            args = [arg for arg in line.split(' ') if arg != '\n' and len(arg) > 0]
            print("Current args:", args)
            out_name = filename[:-3]
            for arg in args:
                out_name += '_'+arg
            callname = 'nvprof -o ' + out_name + ' -f python3' + filename
            try:
                call(['nvprof','-o',out_name+'.nvvp','-f','python3',filename] + args)
                log.write(callname+ ': '+ 'Success')
            except:
                log.write(callname+ ': FAILED')


