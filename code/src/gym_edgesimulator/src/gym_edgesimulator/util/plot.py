from termcolor import colored
import termcolor
from itertools import product
import numpy as np


def plot_color(service_server, servers_mem, services_mem):
    """
        draw the servers placement
        in color
    """
    color = list(termcolor.COLORS.keys())
    color.remove('grey')
    styles = list('■⬤░')
    indicators = list(product(styles, color))
    service_styles = []
    for server, mem in enumerate(servers_mem):
        formatting = '{}'
        values = ''
        services = np.where(server==service_server)[0]
        tot = 0
        meta_chrs = 0
        filled = False
        for s in sorted(services):
            val_fmt = '{:'+indicators[s][0]+'^%d}'%services_mem[s]
            txt = val_fmt.format(str(s))
            if tot + len(txt) > mem and not filled:
                filled = True
                txt = txt[:mem-tot] + ']' + txt[mem-tot:]
            values += colored(txt, indicators[s][1], 'on_grey')
            meta_chrs += 1
            service_styles.append(colored('sv %02d'%s + indicators[s][0], indicators[s][1], 'on_grey'))
            tot += services_mem[s]
        values = formatting.format(values)
        values = ('server %02d [' + values +\
                ("-"*(mem - tot) if (mem-tot) > 0 else "" )+\
                (']' if tot < mem else ''))%server
        print(values)
    print(' '.join(sorted(service_styles, key=lambda x: int(x.split(' ')[1][:2]))))



def plot_black(service_server, servers_mem, services_mem):
    """
        draw the servers placement
        in black and white
    """
    color = list(termcolor.COLORS.keys())
    color.remove('grey')
    styles = list('■⬤░')
    indicators = list(product(styles, color))
    service_styles = []
    style_toggle = 0
    for server, mem in enumerate(servers_mem):
        formatting = '{}'
        values = ''
        services = np.where(server==service_server)[0]
        tot = 0
        meta_chrs = 0
        filled = False
        for s in sorted(services):
            style_toggle += 1
            st = styles[0] if style_toggle%2==0 else styles[2]
            val_fmt = '{:'+st+'^%d}'%services_mem[s]
            txt = val_fmt.format(str(s))
            if tot + len(txt) > mem and not filled:
                filled = True
                txt = txt[:mem-tot] + ']' + txt[mem-tot:]
            values += txt
            meta_chrs += 1
            service_styles.append('sv %02d'%s+st)
            tot += services_mem[s]
        values = formatting.format(values)
        values = ('server %02d [' + values +\
                ("-"*(mem - tot) if (mem-tot) > 0 else "" )+\
                (']' if tot < mem else ''))%server
        print(values)
    print(' '.join(sorted(service_styles, key=lambda x: int(x.split(' ')[1][:2]))))