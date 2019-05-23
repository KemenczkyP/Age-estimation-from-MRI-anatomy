% This script applies the following preprocessing steps to the T2*-weighted
% resting-state functional images: motion correction (set 'do_realign' to
% 1 and 'do_copy' to 1), structural-functional coregistration (set 'do_coreg'
% to 1), spatial normalization to MNI space (set 'do_normalize' to 1 first; 
% then set 'do_norm_write' to 1 after coregistration has been done) spatial 
% smoothing with a full-width half maximum of 8 mm (set 'do_smooth' to 1; 
% fwhm = 8). Set all the values of the variables below accordingly.
% Requires SPM12: www.fil.ion.ucl.ac.uk/spm/software/spm12/

clear all

%% Basic settings (including path variables)

basename = 'sald';
exptype = 'rest';
rootpath = 'D:\Peti\Aging\data\mig_v2_overusers\';        % Study path
rawnifti_folder = [rootpath 'files\'];  % Path to raw nifti files.
fls = dir(sprintf('%s*.nii',rawnifti_folder));

%% Initializing SPM

spmfolder = 'E:\spm12';                % Path to SPM.
addpath(spmfolder)
codefolder = pwd;                      % Current folder containing the codes.
spm('Defaults','FMRI')
spm_jobman('initcfg')
rmpath('E:\spm12\toolbox\OldNorm');    % Removing folder from SPM path.
rmpath('E:\spm12\toolbox\OldSeg');     % Removing folder from SPM path.

%% Preprocessing

% Preprocessing parameters

do_realign = 1;                                    % Motion correction.
refsli = 1;                                         % Reference slice for motion correction.
do_copy = 1;                                      % Coregistration.
do_normalize = 1;                                   % Calculating spatial normalization parameters.
do_norm_write = 1;                                  % Spatial normalization.
tonormwrite_prefix = 'cr'; 
do_smooth = 0;                                      % Spatial smoothing.
fwhm = 8;                                           % Full-width half maximum.
tosmooth_prefix = 'wcr';

% Performing the preprocessing

for subnum=1:length(fls)

    fname=fls(subnum).name;
    
    subname = (strsplit(fname,'.'));
    subname = subname{1,1};
    
    %subname=fname(regexp(fname,'\d'));
    
    spmfolder=sprintf('%sSALD_spm\\%s\\',rootpath,subname);
    processfolder=sprintf('%sprocess\\',spmfolder);
    anatfolder=sprintf('%sSALD_spm\\%s\\anat\\',rootpath,subname);
    anafile=sprintf('%s.nii',subname);
    corename=sprintf('%s',subname);
    
    if ~isdir(processfolder)
        mkdir(processfolder)
    end
    if ~isdir(anatfolder)
        mkdir(anatfolder)
    end
    paramfile=fullfile(processfolder,'nifti_params.mat');
    rawfnames=[];
    dynums=[];
    slinum=[];
    if ~exist(paramfile)
        ht_act=spm_vol(fullfile(rawnifti_folder,fname));
        dynums=length(ht_act);
        slinum=ht_act(1).dim(3);
        rawfname=fname(1:end-4);
        ht=ht_act(1);
        %TR=ht(1).private.timing.tspace;
        save(paramfile,'dynums','slinum','ht','rawfname')%,'TR')
        clear ht ht_act
    else
        load(paramfile)
    end
    
    if do_copy
        tempbatch=[];
        infile=sprintf('%s%s.nii',rawnifti_folder,rawfname);
        outfile=sprintf('%s%s.nii',processfolder,corename);
        copyfile(infile,outfile)
        infile=sprintf('%s%s',rawnifti_folder,anafile);
        outfile=sprintf('%s%s',anatfolder,anafile);
        copyfile(infile,outfile)
    end
    if do_normalize
        tempbatch=[];
        tpmdir=[spm('Dir') '\tpm'];
        tempbatch{1}.spm.spatial.preproc.channel.vols = {fullfile(anatfolder,anafile)};
        tempbatch{1}.spm.spatial.preproc.channel.write = [0 1];
        ngaus  = [2 2 2 3 4 2];
        native = [1 1 1 0 0 0];
        for c = 1:6 % tissue class c
            tempbatch{1}.spm.spatial.preproc.tissue(c).tpm = {
                fullfile(tpmdir, sprintf('TPM.nii,%d', c))};
            tempbatch{1}.spm.spatial.preproc.tissue(c).ngaus = ngaus(c);
            tempbatch{1}.spm.spatial.preproc.tissue(c).native = [native(c) 0];
            if c < 4
                tempbatch{1}.spm.spatial.preproc.tissue(c).warped = [1 1];
            else
                tempbatch{1}.spm.spatial.preproc.tissue(c).warped = [0 0];
            end
        end
        tempbatch{1}.spm.spatial.preproc.warp.mrf = 1;
        tempbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
        tempbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
        tempbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
        tempbatch{1}.spm.spatial.preproc.warp.fwhm  = 0;
        tempbatch{1}.spm.spatial.preproc.warp.samp = 3;
        tempbatch{1}.spm.spatial.preproc.warp.write = [1 1];
        spm_jobman('run',tempbatch)
        cortexfile=['c1' anafile];
        spm_imcalc(fullfile(anatfolder,cortexfile),fullfile(anatfolder,['thr_' cortexfile]),'i1>0.1')
    end
    
    if do_norm_write
        deffield_file=fullfile(anatfolder,['y_' anafile]);
        
        %%anat
        tempbatch=[];
        tempbatch{1}.spm.spatial.normalise.write.subj.resample{1,1} = fullfile(anatfolder,['m' anafile]);
        tempbatch{1}.spm.spatial.normalise.write.subj.def = {deffield_file};
        tempbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
            78 76 85];
        tempbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
        tempbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
        spm_jobman('run',tempbatch)
        
        %%funci
        tempbatch=[];
        for i=1:dynums
            tempbatch{1}.spm.spatial.normalise.write.subj.resample{i,1} = ...
                sprintf('%s\\%s.nii,%g',processfolder,corename,i);
        end
        tempbatch{1}.spm.spatial.normalise.write.subj.def = {deffield_file};
        tempbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
            78 76 85];
        tempbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
        tempbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
        spm_jobman('run',tempbatch)
       
    end

    try
        clear tempbatch
    catch
    end
    cd(codefolder)
    
end