import os
import re

IMAGEJ_EXECUTABLE_LOCATION = r'C:\"Program Files"\fiji-win64\Fiji.app\ImageJ-win64.exe'
PREPROCESS_DIRECTORY = r'C:\Users\Dell\Documents\LabView_FISH_PKU\FISH_analysis_pipeline\lib\preprocess'
def run_ijm(file):
    os.system(IMAGEJ_EXECUTABLE_LOCATION + f' --headless -macro --mem=32000M "{os.path.join(PREPROCESS_DIRECTORY,file)}"')
    #os.system(IMAGEJ_EXECUTABLE_LOCATION + f' --headless --console -macro --mem=32000M "{os.path.join(PREPROCESS_DIRECTORY,file)}"')

def create_mist_command(tile_x,tile_y,input_dir,output_dir,glob_file=None):
    with open(os.path.join(PREPROCESS_DIRECTORY,'MIST_test.ijm')) as f:
        t = f.read()
        t = re.sub('imagedir=.* filenamepattern=',f'imagedir=[{input_dir}] filenamepattern=',t)
        t = re.sub('outputpath=.* display',f'outputpath=[{output_dir}] display',t)
        t = re.sub('gridwidth=.* gridheight=.* starttile',f'gridwidth={tile_x} gridheight={tile_y} starttile',t)
        t = re.sub('extentwidth=.* extentheight=.* timeslices=',f'extentwidth={tile_x} extentheight={tile_y} timeslices=',t)
    with open(os.path.join(PREPROCESS_DIRECTORY,'MIST_temp.ijm'),'w') as f:
        f.write(t)
    
if __name__ == "__main__":
    #create_mist_command(5,6,'something_in','something_out')
    run_ijm('./MIST_temp.ijm')