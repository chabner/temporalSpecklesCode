mexcuda('MatlabInterface.cu','-R2018a','-output','sstmc2D','-DDIMS=2','-lcublas');
mexcuda('MatlabInterface.cu','-R2018a','-output','sstmc3D','-DDIMS=3','-lcublas');
