import numpy as np 
import matplotlib.pyplot as plot


def get_runtimes_davinci():
    # return runtimes in MICROSECONDS
    runtimes = {} 
    descriptors = {}
    # softmax 
    softmax = {}
    softmax['baseline'] = 378
    softmax['fully fused'] = 100
    softmax['partially fused'] = 145
    softmax['pytorch reference'] = 58
    descriptors['softmax'] = 'Float32, 16 x 16 x 128 x 128 (Davinci)'
    runtimes['softmax'] = softmax 

    # vadv
    vadv = {}
    vadv['baseline'] = 816
    vadv['fully fused'] = 157
    vadv['partially fused'] = 430 
    descriptors['vadv'] = 'Float32, 128 x 128 x 80 (Davinci)'
    runtimes['vadv'] = vadv

    # hdiff1 
    hdiff_mini = {} 
    hdiff_mini['baseline'] = 0
    hdiff_mini['fully fused'] = 0
    descriptors['hdiff_mini'] = 'Float32, 128 x 128 x 80 (Davinci)'
    runtimes['hdiff_mini'] = hdiff_mini


    # hdiff2
    hdiff = {}
    hdiff['baseline'] = 0
    hdiff['partially fused'] = 0
    hdiff['fully fused'] = 0
    descriptors['hdiff'] = 'Float32, 128 x 128 x 80 (Davinci)'
    runtimes['hdiff'] = hdiff 


    return runtimes, descriptors


def plot_runtimes(name):
    runtimes, descriptors = get_runtimes_davinci()
    plot_dict = runtimes[name]
    descriptor = descriptors[name]
    x_label, y = [], [] 
    for xl_i, y_i in sorted(plot_dict.items(), key = lambda a: a[1], reverse = True):
        x_label.append(xl_i)
        y.append(y_i) 
    
    #x_label, y = plot_dict.items()
    N = len(plot_dict)
    x = np.arange(N)


    rects = plot.bar(x, y, width = 0.5, color = 'green')
    plot.xticks(x, x_label)
    #plot.yticks()
    plot.ylabel("Runtime [qs]")
    print("DESCRIPTOR=", descriptor)
    plot.suptitle(name.capitalize())
    plot.title(descriptor)
    max_height = rects[0].get_height()
    for rect in rects:
        height = rect.get_height()
        plot.annotate(
            '{}'.format(height),
            xy = (rect.get_x() + rect.get_width() / 2, height),
            xytext = (0,-15 if max_height - height < 30 else 3),
            textcoords = 'offset points',
            ha = 'center',
            va = 'bottom')

    plot.show()

plot_runtimes('softmax')

