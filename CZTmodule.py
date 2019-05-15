import matplotlib.pyplot as plt
import csv
from matplotlib.colors import LogNorm
import numpy as np

def getAveragePixelValue(img, pixels,weight_mask):
    '''
    Averages the dead pixel using the 8 nearest neighbours
    :param img: the projection image
    :param pixels: the problem pixels (is a 2-tuple)
    :return:
    '''
    x,y = pixels
    yborder, xborder = np.shape(img)

    n1 = img[y+1,x] * weight_mask[y+1,x] if (y+1 < yborder) else 0
    n2 = img[y-1,x] * weight_mask[y-1,x] if (y-1 >= 0) else 0
    n3 = img[y,x+1] * weight_mask[y,x+1] if (x+1 < xborder) else 0
    n4 = img[y,x-1] * weight_mask[y,x-1] if (x-1 >= 0) else 0
    n5 = img[y+1,x+1] * weight_mask[y+1,x+1] if (y+1 < yborder and x+1 < xborder) else 0
    n6 = img[y-1,x+1] * weight_mask[y-1,x+1] if (y-1 >= 0 and x+1 < xborder) else 0
    n7 = img[y+1,x-1] * weight_mask[y+1,x-1] if (y+1 < yborder and x-1 >= 0) else 0
    n8 = img[y-1,x-1] * weight_mask[y-1,x-1] if (y-1 >= 0 and x-1 >= 0) else 0
    all_pixels = np.array([n1,n2,n3,n4,n5,n6,n7,n8])

    if np.sum(all_pixels) == 0:
        avg = 0
    else:
        avg = np.average(all_pixels[all_pixels>0])
    return avg


def removeBadPixels(img,mask):
    '''
    The mask is for across all energy levels
    :param img:
    :param mask:
    :return:
    '''
    listpix_y, listpix_x = np.where(mask == 1)

    weight_mask = mask != 1
    for p in np.arange(np.size(listpix_y)):
        piy,pix = listpix_y[p], listpix_x[p]
        newpixelvalue = getAveragePixelValue(img,(pix,piy),weight_mask.astype(int))
        img[piy,pix] = newpixelvalue
    return img


def generateImageByReadingCSVdata(directory,filename, energyname = "EC"):
    '''
    Reads in the file to produce a projection image from the csv data
    :param directory:
    :param filename:
    :param energyname: one of 'EC', 'SEC0-5'
    :return:
    '''
    energy_dict = {'SUMCC': 18,
                   'CC0' :  11,
                   'CC1' : 10,
                   'CC2' : 9,
                   'CC3' : 8,
                   'CC4' : 7,
                   'CC5' : 6,
                   'EC' : 5,
                   'SEC5' : 12,
                   'SEC4' : 13,
                   'SEC3' : 14,
                   'SEC2' : 15,
                   'SEC1' : 16,
                   'SEC0' : 17
                   }
    rows = 24
    columns = 36
    old_pixel_module = -1
    projection_image = np.zeros([rows, columns])
    with open(directory+filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for csv_row in spamreader:
            current_pixel_module = csv_row[0]
            if current_pixel_module != "Pixel":

                row = int(csv_row[2])
                column = int(csv_row[3])
                if energyname != 'Kedge':
                    entry = np.float(csv_row[energy_dict[energyname]])
                else:
                    #This should be unreachable. This was when I did the K-edge subtraction earlier
                    cc4entry = np.float(csv_row[energy_dict['CC4']])
                    cc3entry = np.float(csv_row[energy_dict['CC3']])
                    entry = cc4entry - cc3entry
                if old_pixel_module != current_pixel_module:
                    projection_image[row,column] += entry
                old_pixel_module = current_pixel_module
    return projection_image

def clickAwayTheBadPixels(energy_name, directory_air, air_files, darkfield_file, directory_dataproj, proj_files, size, window, save=True):
    dp_mask = np.zeros(size)

    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        indx, indy = round(event.xdata), round(event.ydata)
        dp_mask[indy,indx] = 1

    #look at all air scans, click away the bad pixels
    for af in air_files:
        air_image = generateImageByReadingCSVdata(directory_air, af, energyname=energy_name)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(air_image, norm=LogNorm(vmin=1e2, vmax=1e7), interpolation="none")
        cbar = fig.colorbar(cax)
        plt.title("open field: energy bin " + energy_name +"\n"+af)
        if save:
            plt.savefig(directory_air + "CZT_"+af[0:-4]+"_"+energy_name + ".png")
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    #look at dark field, click away the bad pixels
    dark_image = generateImageByReadingCSVdata(directory_air,darkfield_file,energyname=energy_name)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(dark_image, norm=LogNorm(vmin=window[0], vmax=window[1]),interpolation="none")
    cbar = fig.colorbar(cax)
    plt.title("dark field projection: energy bin " + energy_name)
    if save:
        plt.savefig(directory_air+"CZT_darkfield_"+energy_name+".png")
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    #We need to correct the dark field since we are subtracting it off
    dark_image_to_subt = removeBadPixels(dark_image,dp_mask)
    new_proj_images = np.zeros(len(proj_files),dtype=object)
    for pf in np.arange(0,len(proj_files)):
        proj_image = generateImageByReadingCSVdata(directory_dataproj, proj_files[pf], energyname=energy_name)
        new_img = removeBadPixels(proj_image,dp_mask)
        new_img = new_img - dark_image_to_subt #subtract the corrected dark field from the corrected proj image
        new_proj_images[pf] = new_img
        plt.imshow(new_img)
        plt.show()

    return new_proj_images