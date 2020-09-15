from subprocess import call 
import sys 


filename = sys.argv[1]
with open('config.txt', 'r') as config:
    with open('log.txt','w') as log:
        line = config.readline()
        args = line.split(' ')
        while line:
            out_name = filename
            for arg in args:
                out_name += '_'+arg
            callname = 'nvprof -o ' + out_name + ' -f python3' + filename
            try:
                call(['nvprof','-o',out_name,'-f','python3',filename])
                log.write(callname, ':', 'Success')
            except:
                log.write(callname,'FAILED')


