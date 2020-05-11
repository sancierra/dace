""" SDFG visualizer that uses Flask, HTML5, and Javascript. """

import json
import tempfile
import sys
import os
import platform

import dace
import diode
import tempfile
import jinja2


def view(sdfg, roofline, filename=None):
    """View an sdfg in the system's HTML viewer
       :param sdfg: the sdfg to view, either as `dace.SDFG` object or a json string
       :param filename: the filename to write the HTML to. If `None`, a temporary file will be created.
    """
    if type(sdfg) is dace.SDFG:
        old_meta = dace.serialize.JSON_STORE_METADATA
        dace.serialize.JSON_STORE_METADATA = False
        sdfg = dace.serialize.dumps(sdfg.to_json())
        dace.serialize.JSON_STORE_METADATA = old_meta

    basepath = os.path.dirname(os.path.realpath(diode.__file__))
    template_loader = jinja2.FileSystemLoader(
        searchpath=os.path.join(basepath, 'templates'))
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template('sdfv_roofline.html')

    # prepare roofline
    save_folder = basepath + '/roofline_cache/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_name = 'tmp'
    print("SAVE NAME=", save_folder + save_name)
    roofline.plot(save_path = save_folder, save_name = save_name)

    html = template.render(sdfg=json.dumps(sdfg), dir=basepath + '/', rooflinedir = save_folder + save_name + '.png')

    if filename is None:
        fd, html_filename = tempfile.mkstemp(suffix=".sdfg.html")
    else:
        fd = None
        html_filename = filename + ".html"

    with open(html_filename, "w") as f:
        f.write(html)

    print("File saved at %s" % html_filename)

    system = platform.system()

    if system == 'Windows':
        os.system(html_filename)
    elif system == 'Darwin':
        os.system('open %s' % html_filename)
    else:
        os.system('xdg-open %s' % html_filename)

    if fd is not None:
        os.close(fd)
