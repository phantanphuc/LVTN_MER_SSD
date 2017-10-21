"""
    DPRL Math Symbol Recognizers 
    Copyright (c) 2012-2014 Kenny Davila, Richard Zanibbi

    This file is part of DPRL Math Symbol Recognizers.

    DPRL Math Symbol Recognizers is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DPRL Math Symbol Recognizers is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DPRL Math Symbol Recognizers.  If not, see <http://www.gnu.org/licenses/>.

    Contact:
        - Kenny Davila: kxd7282@rit.edu
        - Richard Zanibbi: rlaz@cs.rit.edu 
"""
import os
import sys
import fnmatch
import string
from traceInfo import *
from mathSymbol import *
from load_inkml import *
import pdb 
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import word2ID
#=====================================================================
#  generates a training set from a directory containing the inkml
#
#  Created by:
#      - Kenny Davila (Oct, 2012)
#  Modified By:
#      - Kenny Davila (Oct 19, 2012)
#      - Kenny Davila (Nov 25, 2013)
#         - Added ID to symbol
#         - Added AUX file to output origin for each sample in DS
#      - Kenny Davila (Jan 16, 2012-2014)
#         - Print number of attributes found
#      - Kenny Davila (March 2016)
#         - Added additional error handling for files with errors
#
#=====================================================================
def normIndex(index,res):
    if index < 300:
        res.append(index)
#        return index
    else:
        normIndex(index-1,res)
    return res
def main():
    #usage check
    if len(sys.argv) != 4:
        print("Usage: python get_training_set.py inkml_path output [train|test]")
        print("Where")
        print("\tinkml_path\t= Path to directory that contains the inkml files")
        print("\toutput\t\t= File name of the output file")
        print("\ttrain|test\t\t= for train or test purpose")
        return
    
    #load and filter the list of files, the result is a list of inkml files only
    try:
        complete_list = os.listdir(sys.argv[1])
        filtered_list = []
        for file in complete_list:
            if fnmatch.fnmatch(file, '*.inkml'):
                filtered_list.append( file )
    except:
        print( "The inkml path <" + sys.argv[1] + "> is invalid!" )
        return

    samples = []
    labels_found = {}
    sources = []
    error_files = []
    sym_num = 0
    w2i, i2w = word2ID.buildVocab('./mathsymbolclass.txt')
    #read every file in the path specified...        
    for i in range(len(filtered_list)):
        file_name = filtered_list[i]
        file_path = sys.argv[1] + '//' + file_name;
        advance = float(i) / len(filtered_list)
        print(("Processing => {:.2%} => "  + file_path).format( advance ))

        try:
            symbols = load_inkml( file_path, True )            
        except:
            print("Failed processing: " + file_path)
            error_files.append(file_path)
            symbols = []
        
        for new_symbol in symbols:
            #now generate the features and add them to the list, including the tag
            #for the expected class....
            sample = new_symbol.getFeatures() + [ new_symbol.truth ]        
#            sample = new_symbol.getSquaredBoundingBox() + [ new_symbol.truth ]
            samples.append( sample )
#            pdb.set_trace()
            #count samples per class
            if not new_symbol.truth in labels_found:
                labels_found[ new_symbol.truth ] = 1
            else:
                labels_found[ new_symbol.truth ] += 1
            #write symbol image
#            x_list = [x for x,y in new_symbol.getPoints()]
#            y_list = [y for x,y in new_symbol.getPoints()]
#            sym_img1 = np.zeros((300,300))
###            pdb.set_trace()
#            for i in range(len(x_list)):
#                x = normIndex(int(round(x_list[i])),[])
#                y = normIndex(int(round(y_list[i])),[])
#                print('x',x)
#                print('y',y)
#                print(i)
#                if type(x[0])!=int or type(y[0]) != int or x[0] >=300 or y[0] >=300:
#                    print('x',x[0])
#                    print('y',y[0])
#                    pdb.set_trace()
#                else:
##                    cv2.line(sym_img, x,y,255, 2)
#                    sym_img1[y[0],x[0]] = 255

#            sym_img = np.zeros((260,260))
#            sym_img = np.zeros( (260,260,3), dtype=np.uint8)
            sym_img = np.zeros( (260,260), dtype=np.uint8)
            points = new_symbol.getPoints()
            for i1 in range(len(points)):
                x_norm = normIndex(int(round(points[i1][0])),[])
                y_norm = normIndex(int(round(points[i1][1])),[])
#                pdb.set_trace()
                points[i1] = (x_norm[0],y_norm[0])
            min_x,max_x,min_y,max_y = new_symbol.getSquaredBoundingBox()
            min_x = normIndex(int(round(min_x)),[])[0]
            max_x = normIndex(int(round(max_x)),[])[0]
            min_y = normIndex(int(round(min_y)),[])[0]
            max_y = normIndex(int(round(max_y)),[])[0]
            bb_points=[(min_x,min_y),(max_x,min_y),(max_x,max_y),(min_x,max_y)]
            # draw symbol
            for i2 in range(len(points)-1):
#                cv2.line(sym_img, points[i2], points[i2+1],[255,255,255],2)
                cv2.line(sym_img, points[i2], points[i2+1],255,2)
            #draw bounding box
#            for i in range(len(bb_points)-1):
#                cv2.line(sym_img, bb_points[i], bb_points[i+1],255,2)
#            cv2.line(sym_img, bb_points[3], bb_points[0],255,2)
            #zeros padding 
#            output = cv2.copyMakeBorder(sym_img, 20,20,20,20,cv2.BORDER_CONSTANT,value = [0,0,0])
            output = cv2.copyMakeBorder(sym_img, 20,20,20,20,cv2.BORDER_CONSTANT,value = 0)
#            plt.imshow(sym_img1.astype(np.uint8), interpolation='bicubic')
#            im = Image.fromarray(sym_img.astype(np.uint8))
#            im.show()
            #output_path =  './trainData/'+file_name[:-6]+'_'+ str(new_symbol.id)+'.png'
#            pdb.set_trace()
            
            #print(output.shape)
            #quit()

            if sys.argv[3] == 'train':
                cv2.imwrite('./../dataset/train/'+str(sym_num+1)+'_' +file_name[:-6]+'_'+ str(new_symbol.id)+'.png', output)
            else:
                cv2.imwrite('./../dataset/test/'+str(sym_num+1)+'_' +file_name[:-6]+'_'+ str(new_symbol.id)+'.png', output)
            
            #cv2.imshow('s', sym_img.astype(np.uint8))
            #cv2.imshow('s', output.astype(np.uint8))
            #cv2.waitKey()
#            plt.show()
#            break
            #the source of current symbol will be
            #exported as auxiliary file
            sources.append( ( str(sym_num+1)+'_' + file_name[:-6]+'_'+ str(new_symbol.id)+'.png',1, (min_x,min_y,max_x,max_y), w2i[new_symbol.truth]) )
            sym_num+=1

    print("Total input files: " + str(len(filtered_list)))
    print("Total valid files: " + str(len(filtered_list) -  len(error_files)))
    print("Files with errors: ")
    for filename in error_files:
        print("\t- " + filename)
        
    print("Total files with error: " + str(len(error_files)))

    print( "Found: " + str(len(labels_found.keys())) + " different classes" )
    if len(samples) > 0:
        print( "Found: " + str(len(samples[0]) - 1 ) + " different attributes" )

    print ("Saving main .... ")
    #now that all the samples have been collected, write them all 
    #in the output file
    try:
        file = open(sys.argv[2], 'w')
    except:
        print( "File <" + sys.argv[2] + "> could not be created")
        return
    
    content = ''
    #print as headers the types for each feature...
    feature_types = new_symbol.getFeaturesTypes()
    for i, feat_type in enumerate(feature_types):
        if i > 0:
            content += '; '
        content += feat_type
    content += '\r\n'   
        
    for sample in samples:
        line = ''
        for i, v in enumerate(sample):
            if i > 0:
                line += '; '
            
            if v.__class__.__name__ == "list":
                #multiple values...
                for j, sv in enumerate(v):
                    if j > 0:
                        line += '; '
                    line += str(sv)
            else:
                #single value...
                line += str(v)
                
        line += '\r\n'                     
        content += line
        
        if len(content) >= 50000:
            file.write(content)
            content = ''
        
    file.write(content)
    
    file.close()

    print ("Saving auxiliary.... ")

    #Now, add the auxiliary file
    try:
        aux_file = open(sys.argv[2] + ".sources.txt" , 'w')
    except:
        print( "File <" + sys.argv[2]  + ".sources.txt> could not be created")
        return

    content = ''
    for source_path,sym_num, sym_bb,sym_truth in sources:
        content += source_path + ' '+str(sym_num)+' '+str(sym_bb[0])+ ' '+str(sym_bb[1]) + ' '+str(sym_bb[2])+ ' '+str(sym_bb[3])+ ' '+str(sym_truth)+'\n'

    aux_file.write(content)

    aux_file.close()

    print ("Done!")
main()
