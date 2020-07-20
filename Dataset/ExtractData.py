import sys
# sys.path.append('/Users/agupta/opt/anaconda3/lib/python3.7/site-packages')
import wget
from pathlib import Path
from itertools import islice
import os
import os.path
from os import path
import urllib3


def bar_progress(current,total,width=80):
    progress_message="Downloading: %d%% [%d / %d] bytes" % (current / total * 100,current,total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


space='    '
branch='│   '
tee='├── '
last='└── '


def tree(dir_path: Path,level: int = -1,limit_to_directories: bool = False,length_limit: int = 1000):
    """Given a directory Path object print a visual tree structure"""
    dir_path=Path(dir_path)  # accept string coerceable to Path
    files=0
    directories=0

    def inner(dir_path: Path,prefix: str = '',level=-1):
        nonlocal files,directories
        if not level:
            return  # 0, stop iterating
        if limit_to_directories:
            contents=[d for d in dir_path.iterdir() if d.is_dir()]
        else:
            contents=list(dir_path.iterdir())
        pointers=[tee] * (len(contents) - 1) + [last]
        for pointer,path in zip(pointers,contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories+=1
                extension=branch if pointer == tee else space
                yield from inner(path,prefix=prefix + extension,level=level - 1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files+=1

    print(dir_path.name)
    iterator=inner(dir_path,level=level)
    for line in islice(iterator,length_limit):
        print(line)
    if next(iterator,None):
        print(f'... length_limit, {length_limit}, reached, counted:')
    print(f'\n{directories} directories' + (f', {files} files' if files else ''))


def create_dir():
    # detect the current working directory and print it
    direc=os.getcwd()
    print("\n\nThe current working directory is %s" % path)
    print('\n\nBefore directory creations\n\n')
    print(tree(direc),1)
    br_file=direc + '/Data/br_file'
    vr_file=direc + '/Data/vr_file'
    br002_file=direc + '/Data/br002_file'
    vr002_file=direc + '/Data/vr002_file'

    if (not path.exists(br_file)):
        os.mkdir(br_file)
        print('\n\nDIRECTORY CREATED')
    if (not path.exists(vr_file)):
        os.mkdir(vr_file)
        print('DIRECTORY CREATED')
    if (not path.exists(br002_file)):
        os.mkdir(br002_file)
        print('DIRECTORY CREATED')
    if (not path.exists(vr002_file)):
        os.mkdir(vr002_file)
        print('DIRECTORY CREATED')
    else:
        print("\n\nDIRECTORIES ALREADY PRESENT\n\n")

    print(tree(direc))


def exists(path):
    try:
        deadLinkFound=True
        http=urllib3.PoolManager()
        r=http.request('GET',path)
        response=r.status
        if (response == 200):
            deadLinkFound=False
            return deadLinkFound
        else:
            return deadLinkFound
    except:
        deadLinkFound=True
        return deadLinkFound


def collectURLs(start,end):
    sources=['hmi_masp_mas_std_0201']

    usefulurl=[]
    for i in range(start,end):
        for src in sources:
            url='http://www.predsci.com/data/runs/cr' + str(i) + '-medium/' + src + '/'
            # print('---------------------------------------------')
            # print('URL is: ',url)
            deadlink=exists(url)
            if not deadlink:
                print('If url exists: True')
                print(url)
                usefulurl.append(url)  # print('---------------------------------------------')
            else:
                # print('If url exists: False')
                dummy=[]
    return usefulurl


def downloadFiles(usefulurl):
    print("\n\nTOTAL NUMBER OF URLS:",len(usefulurl))
    countleft=len(usefulurl)
    direc=os.getcwd()
    if (len(usefulurl) > 0):
        for url in usefulurl:
            urlbr=url + "helio/br_r0.hdf"
            print("\n",urlbr)
            wget.download(urlbr,'./Data/br_file',bar=bar_progress)
            os.rename(direc + "/Data/br_file/br_r0.hdf",direc + "/Data/br_file/br_r0_" + str(len(usefulurl) - countleft) + ".hdf")

            urlvr=url + "helio/vr_r0.hdf"
            print("\n",urlvr)
            wget.download(urlvr,'./Data/vr_file',bar=bar_progress)
            os.rename(direc + "/Data/vr_file/vr_r0.hdf",direc + "/Data/vr_file/vr_r0_" + str(len(usefulurl) - countleft) + ".hdf")

            urlbr2=url + "helio/br002.hdf"
            print("\n",urlbr2)
            wget.download(urlbr2,'./Data/br002_file',bar=bar_progress)
            os.rename(direc + "/Data/br002_file/br002.hdf",
                      direc + "/Data/br002_file/br002_" + str(len(usefulurl) - countleft) + ".hdf")

            urlbr2=url + "helio/vr002.hdf"
            print("\n",urlbr2)
            wget.download(urlbr2,'./Data/vr002_file',bar=bar_progress)
            os.rename(direc + "/Data/vr002_file/vr002.hdf",
                      direc + "/Data/vr002_file/vr002_" + str(len(usefulurl) - countleft) + ".hdf")

            print("\n")
            countleft=countleft - 1
            print(countleft,": URLS LEFT")
        print("\n\nDOWNLOAD COMPLETED !!!")


def allSamples():
    create_dir()
    start=1625
    end=2226
    downloadFiles(collectURLs(start,end))


def singleSample():
    create_dir()
    start=2225
    end=2226
    downloadFiles(collectURLs(start,end))


def nNumberSamplesBeg():
    val=input("\n\nEnter number of Samples you need: ")
    create_dir()
    start=1625
    end=1625 + int(val)
    downloadFiles(collectURLs(start,end))


def nNumberLatestSamples():
    val=input("\n\nEnter number of Latest Samples you need: ")
    create_dir()
    start=2225 - int(val) + 1
    end=2226
    downloadFiles(collectURLs(start,end))


switcher={1:allSamples,2:singleSample,3:nNumberSamplesBeg,4:nNumberLatestSamples
}


def numbers_to_strings(argument):
    # Get the function from switcher dictionary
    func=switcher.get(argument,"nothing")
    # Execute the function
    return func()


print("ENTER THE NUMBER OF SAMPLES YOU NEED :\n")
print("Options are :\n")
print("1 : All the samples")
print("2 : Single sample")
print("3 : n number of samples from beginning")
print("4 : n number of latest samples")
val=input("\n\nEnter your option (1,2,3,4): ")
numbers_to_strings(int(val))