# -*- coding: utf-8 -*-
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.append(dir_path)
import tkinter
import tkinter.filedialog as filedialog

def get_fileloc(title, initialdir):
    """ Obtains the data path of recordings by using a graphical user
    interface (GUI).

    Parameters
    ----------
    title : str
        The title of the GUI window.
    initaldir : str
        The directory GUI initiates. Default is my pathway for the file with
        pData.

    Returns
    -------
    fileloc
        The path to the file chosen by the user through GUI.

    """
    root = tkinter.Tk()
    root.withdraw()
    fileloc = filedialog.askopenfilename(initialdir=initialdir,title=title)
    return fileloc

def get_dirloc(title, initialdir):
    """ Obtains a directory path by using a graphical user
    interface (GUI).

    Parameters
    ----------
    title : str
        The title of the GUI window.
    initaldir : str
        The directory GUI initiates. Default is my pathway for the file with
        pData.

    Returns
    -------
    dirloc
        The path to the directory chosen by the user through GUI.

    """
    root = tkinter.Tk()
    root.withdraw()
    fileloc = filedialog.askdirectory(initialdir=initialdir,title=title)
    return fileloc

def change_line(regex, line, lines, path, i):
    if line.startswith(regex):
        lines[i] = regex + '\'' + path + '\'' + '\n'

# Find the user specific directories
initialdir = dir_path
pdataf = get_dirloc('Select the general experiment folder with pData',
               initialdir)

initialdir = pdataf
savef = get_dirloc('Select a folder for saving python derived data',
               initialdir)

stimpath = get_fileloc('Select the white noise stimulus file (in .mat format)',
               initialdir)

# Write the user specific directories to the script
rfpy = os.path.join(dir_path, 'master_script.py')
with open(rfpy, 'r+') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        # Changing locations in the script
        change_line('pdataf = ', line, lines, pdataf, i)
        change_line('savef = ', line, lines, savef, i)
        change_line('stimpath = ', line, lines, stimpath, i)
        change_line('modulepath = ', line, lines, dir_path, i)
    f.seek(0)
    f.writelines(lines)


