# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:33:23 2019

@author: Jonathan Mushkin (Weizmann Institute of Science)

Functions and object classes, used to load and save files in different formats.
"""
import h5py
import scipy.io as sio

def write_layer(gp, adict):
    """
    taken almost as is from the internet
    recursively writes layer after layer of the object into the h5 file
    """
    for k,v in adict.items():
        if isinstance(v, dict):
            gp1 = gp.create_group(k)
            write_layer(gp1, v)
                              
        else:
            gp.create_dataset(k, data=v)
            

def load_h5(filename, obj ):
    """
    load a .h5 file and save it to an object
    INPUTS:
        filename : file to load the data from
        obj : object with the correct fields to read the data into
    OUTPUT:
        obj : object with fields filled according the the file data
    
    
    NOTE:
        This function is not smart. It has to have an object of the correct 
        fields as input, otherwise it will not work.
        If the file has more keys than the object has fields, the function will 
        fail.
        Otherwise (more field in the onject than keys in the file), it should 
        work properly. BEST TO HAVE PERFECT MATCH
        
    """
    fr = h5py.File(filename,  "r")
    field_names = list(fr.keys())
    
    for name in field_names:
        setattr(obj ,name, fr[name][()])
    fr.close()
    return obj

def save_h5(filename, obj):
    """
    Save object (obj) with given specification (spec) to a h5 file (filename).
    """
    if not(filename.endswith('.h5')):
        filename = filename + '.h5'
    
    attributes = list(obj.__dict__.keys())
    d = {}
    for name in attributes:
        d[name] = getattr(obj, name)
    
    fw = h5py.File(filename,'w')
    write_layer(fw,d)
    fw.close()


def save_mat(filename,obj):
    attributes = list(obj.__dict__.keys())
    d = {}
    for name in attributes:
        d[name] = getattr(obj, name)
        
    sio.savemat(filename,d)

def load_mat(filename):
    d = sio.loadmat(filename)
    obj = obj_dic(d)
    for name in list(obj.__dict__.keys()):
        if not(name.startswith('_')):
            X = getattr(obj,name)
            if X.size==1:
                Y = X.ndim
                for y in range(Y):
                    X = X[0]
                setattr(obj, name, X)
            
    return obj



def obj_dic(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, obj_dic(j))
        elif isinstance(j, seqs):
            setattr(top, i, 
                type(j)(obj_dic(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top
