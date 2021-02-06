import numpy as np 
import matplotlib.pyplot as plot


def get_runtimes_davinci():
    # return runtimes in MICROSECONDS
    runtimes = {} 
    descriptors = {}
    colors = {}
    # softmax 
    softmax = {}
    color = {}
    softmax['baseline'] = 378
    softmax['fully fused'] = 100
    softmax['partially fused'] = 145
    softmax['pytorch reference'] = 58
    descriptors['softmax'] = 'Float32, 16 x 16 x 128 x 128 (Davinci)'
    runtimes['softmax'] = softmax 
    color['default'] = 'green'
    color['pytorch reference'] = 'red'
    colors['softmax'] = color 

    # vadv
    vadv = {}
    color = {}
    vadv['baseline'] = 870
    vadv['fully fused'] = 245
    vadv['partially fused'] = 430 
    vadv['CSCS best bench'] = 280
    descriptors['vertical advection'] = 'Float32, 128 x 128 x 80 (Davinci)'
    runtimes['vertical advection'] = vadv
    color['default'] = 'green'
    color['CSCS best bench'] = 'red'
    colors['vertical advection'] = color


    # hdiff1 
    hdiff_mini = {} 
    color = {}
    hdiff_mini['baseline'] = 340
    hdiff_mini['fully fused'] = 120
    descriptors['horizontal diffusion (mini)'] = 'Float32, 128 x 128 x 80 (Davinci)'
    runtimes['horizontal diffusion (mini)'] = hdiff_mini
    color['default'] = 'green'
    colors['horizontal diffusion (mini)'] = color


    # hdiff2
    hdiff = {}
    color = {}
    hdiff['baseline'] = 540
    hdiff['partially fused'] = 310
    hdiff['fully fused'] = 550
    descriptors['horizontal diffusion'] = 'Float32, 128 x 128 x 80 (Davinci)'
    runtimes['horizontal diffusion'] = hdiff 
    color['default'] = 'green'
    colors['horizontal diffusion'] = color

    return runtimes, descriptors, colors


def plot_runtimes(name):
    runtimes, descriptors, colors = get_runtimes_davinci()
    plot_dict = runtimes[name]
    descriptor = descriptors[name]
    x_label, y, c = [], [], []
    for xl_i, y_i in sorted(plot_dict.items(), key = lambda a: a[1], reverse = True):
        x_label.append(xl_i)
        y.append(y_i) 
        print(colors)
        print(name)
        print(colors[name])
        if xl_i in colors[name]:
            c.append(colors[name][xl_i])
        else:
            c.append(colors[name]['default'])
    
    #x_label, y = plot_dict.items()
    N = len(plot_dict)
    x = np.arange(N)


    rects = plot.bar(x, y, width = 0.5, color = c)
    plot.xticks(x, x_label)
    plot.ylabel("Runtime [qs]")
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
    plot.savefig(name+'.pdf')

plot_runtimes('vertical advection')

