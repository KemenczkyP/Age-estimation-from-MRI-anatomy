import argparse
import os
import sys

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

import matplotlib as mpl
sns.set("talk",
        {'axes.grid' : False})

parser = argparse.ArgumentParser(description='Activation visualization flags')
parser.add_argument('--sample_MRI',       default=os.getcwd()+'\\DATA\\INHOUSE\\wmanat_aging_000004.npy', type=str, help='Path to sample MRI image.')

parser.add_argument('--migraine_heatmap', default=os.getcwd()+'\\heatmaps\\mig_heatmap_small.npy', type=str, help='Average heatmap of the migraine group')
parser.add_argument('--control_heatmap',  default=os.getcwd()+'\\heatmaps\\ctr_heatmap_small.npy', type=str, help='Average heatmap of the control group')

args = parser.parse_args()

data = np.load(args.sample_MRI)
hm_mig = np.load(args.migraine_heatmap)
hm_ctr = np.load(args.control_heatmap)

def normalize(image, type_ = '01'):
    '''
    type_: "01" - between 0 and 1
    '''
    if(type_ == '01'):
        image = image - np.min(image)
        image = image / np.max(image)
    return image


# normalize heatmaps and compute the difference
global_normalized = normalize(np.array([hm_ctr,
                                        hm_mig]))
ctr_heatmap = global_normalized[0,:,:,:]
mig_heatmap = global_normalized[1,:,:,:]
dif_heatmap = mig_heatmap-ctr_heatmap
dif_heatmap = dif_heatmap-dif_heatmap.min()

# matplotlib settings
gridspec.GridSpec(2,21)
ax0 = plt.subplot2grid((2,21), (0,0), colspan=8, rowspan=2)
ax1 = plt.subplot2grid((2,21), (0,8), colspan=4, rowspan=1)
ax2 = plt.subplot2grid((2,21), (0,12), colspan=4, rowspan=1)
ax3 = plt.subplot2grid((2,21), (0,16), colspan=4, rowspan=1)
ax4 = plt.subplot2grid((2,21), (1,8), colspan=4, rowspan=1)
ax5 = plt.subplot2grid((2,21), (1,12), colspan=4, rowspan=1)
ax6 = plt.subplot2grid((2,21), (1,16), colspan=4, rowspan=1)
ax7 = plt.subplot2grid((2,21), (0,20), colspan=1, rowspan=1)
ax8 = plt.subplot2grid((2,21), (1,20), colspan=1, rowspan=1)

plt.subplots_adjust(wspace = 0.7)

# animate plot
for idx in range(0, data.shape[2]):

    # compute mni z-coordinate from index
    MNI_coord = (idx*2)-70

    # noramize sample volume and get the appropriate slices
    example_slice = normalize(data[:,:,idx])
    example_heatmap_ctr = ctr_heatmap[:,:,idx].astype(np.float32)
    example_heatmap_mig = mig_heatmap[:,:,idx].astype(np.float32)
    example_heatmap_dif = dif_heatmap[:,:,idx].astype(np.float32)

    # sample image in its own
    ax0.imshow(example_slice, cmap='Greys')
    ax0.set_title('T1 weighted MRI volume\ntransformed into MNI space\n MNI Z-coordinate: {}\n'.format(MNI_coord))

    # control heatmap
    ax1.imshow(example_heatmap_ctr, vmin=ctr_heatmap.min(), vmax=ctr_heatmap.max())
    ax1.set_title('Averaged heatmap\nfor control group')

    # control heatmap on sample image
    ax4.imshow(example_slice, cmap='Greys')
    ax4.pcolormesh(example_heatmap_ctr, cmap='Purples', alpha = 0.5, vmin=ctr_heatmap.min(), vmax=ctr_heatmap.max())
    ax4.set_title('Heatmap on sample\nMRI volume \n(control)')

    # migraine heatmap
    ax2.imshow(example_heatmap_mig, vmin=ctr_heatmap.min(), vmax=ctr_heatmap.max())
    ax2.set_title('Averaged heatmap\nfor migraine group')

    # migraine heatmap on sample image
    ax5.imshow(example_slice, cmap='Greys')
    ax5.pcolormesh(example_heatmap_mig, cmap='Purples', alpha = 0.5, vmin=ctr_heatmap.min(), vmax=ctr_heatmap.max())
    ax5.set_title('Heatmap on sample\nMRI volume\n(migraine)')

    # mig_heatmap-ctr_heatmap heatmap
    ax3.imshow(example_heatmap_dif, vmin=ctr_heatmap.min(), vmax=ctr_heatmap.max())
    ax3.set_title('Intensity difference \nbetween groups')

    # mig_heatmap-ctr_heatmap heatmap on sample image
    ax6.imshow(example_slice, cmap='Greys')
    ax6.pcolormesh(example_heatmap_dif, cmap='Purples', alpha = 0.5, vmin=ctr_heatmap.min(), vmax=ctr_heatmap.max())
    ax6.set_title('Intensity difference\non brain')

    # remove ticks
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax5.set_yticks([])
    ax5.set_xticks([])
    ax6.set_yticks([])
    ax6.set_xticks([])

    # show color bars
    norm = mpl.colors.Normalize(vmin=ctr_heatmap.min(),
                                vmax=ctr_heatmap.max())
    cb1 = mpl.colorbar.ColorbarBase(ax=ax7,
                                    norm=norm,
                                    orientation='vertical')
    norm = mpl.colors.Normalize(vmin=ctr_heatmap.min(),
                                vmax=ctr_heatmap.max())
    cb1 = mpl.colorbar.ColorbarBase(ax=ax8,
                                    norm=norm,
                                    orientation='vertical',
                                    cmap=mpl.cm.Purples)

    # plt animation
    plt.pause(0.5)
    ax0.cla()
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    ax5.cla()
    ax6.cla()
    ax7.cla()
    ax8.cla()
