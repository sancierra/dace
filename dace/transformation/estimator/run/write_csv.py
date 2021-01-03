import matplotlib.pyplot as plt 
from operator import add 


FILENAME = 'test.csv'
AGGREGATE = False 
MAX_I = 10
size_dict = {'Mbyte': 1, 
             'Kbyte': 0.001,
             'byte' : 0.000001,
             'usecond' : 1,
             'msecond' : 1000,
             }

with open(FILENAME, 'w') as f:
    f.write('id, read, write, sum, runtime\n')
    for i in range(1,MAX_I):
        file_len = 0
        with open(f'output{i}.txt','r') as current:
            for line in current:
                file_len += 1

        with open(f'output{i}.txt','r') as current:
            line = current.readline()[:-1]
            counter = 0
            while counter < file_len:
                if len(line) > 1 and all([len(e) == 0 or all(ee == '-' for ee in e) for e in line.split(' ')]):
                    read = current.readline()[:-1].split(' ')
                    write = current.readline()[:-1].split(' ')
                    duration = current.readline()[:-1].split(' ')
                    _ = current.readline() 
                    counter += 4 

                    x = f'{i}'
                    rwd = [] 
                    for ln in [read, write, duration]:
                        quantity = None
                        # pre-multiplier 
                        for s in ln:
                            if len(s) == 0:
                                continue 
                            if s in size_dict:
                                quantity = size_dict[s]
                            if quantity is not None:
                                try:
                                    quantity *= float(s)
                                    break 
                                except ValueError:
                                    pass 
                        else:
                            raise RuntimeError("Cannot convert row!")
                        
                        rwd.append(quantity)
                    
                    x = f'{i},{rwd[0]},{rwd[1]},{rwd[0]+rwd[1]},{rwd[2]}\n'
                    print("Line =", x )
                    f.write(x)
                line = current.readline()[:-1]
                counter += 1

# create graph 
with open(FILENAME, 'r') as f:
    if AGGREGATE:
        aggregates = dict() 
        for (i,line) in enumerate(f):
            if i == 0:
                continue
            print(line)
            id = int(line.split(',')[0])
            (read,write,sumrw,duration) = [float(e) for e in line.split(',')[1:]]
            if id in aggregates:
                aggregates[id] = list(map(add, aggregates[id], [read, write, sumrw, duration]))
            else:
                aggregates[id] = [read, write, sumrw, duration]
        x, y = [], []
        for (r, w, s, d) in aggregates.values():
            x.append(s)
            y.append(d)
        plt.scatter(x,y)
        plt.show()
    else:
        x, y = [], []
        for (i,line) in enumerate(f):
            if i == 0:
                continue
            print([float(e) for e in line.split(',')[1:]])
            (read,write,sumrw,duration) = [float(e) for e in line.split(',')[1:]]
            x.append(sumrw)
            y.append(duration)
        print(x,y)
        plt.scatter(x,y)
        plt.title('Test')
        plt.savefig('plot.pdf')