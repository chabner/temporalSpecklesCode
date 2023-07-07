focal_x = -50:1:50;
focal_z = 0:2:20;
tau_x = -1:1:1;

%% Configuration

% Simulation
config.simulation.batchSize = 50;
config.simulation.deviceNumber = 0;
config.simulation.fullIteration = 100;
config.simulation.lambda = 1;


config.scattering.type = repmat({'vMF'},1,5);
config.scattering.amplitudeFunction = cell(1,5);

% Isotropig rendering
config.scattering.amplitudeFunction{1}.mixtureMu = 0;
config.scattering.amplitudeFunction{1}.mixtureC = -1.2655121234846453964889;
config.scattering.amplitudeFunction{1}.mixtureAlpha = 1;

% ~ HG g = 0.95
config.scattering.amplitudeFunction{2}.mixtureMu = [0;7.864623088553429;1.432699943392017;34.940513109419960;1.582028646657162e+02;7.527763768351407e+02;-0.026667703987043];
config.scattering.amplitudeFunction{2}.mixtureC = [-2.531024246969291;-7.640125542225978;-2.852369324003370;-33.224743185787034;-1.549768635690816e+02;-7.479904856941710e+02;-2.531142771898976];
config.scattering.amplitudeFunction{2}.mixtureAlpha = [0.096076950725656;0.250010487228936;0.504158076691359;0.163528156678287;0.097875866684034;0.031641230957240;0.223222884401916];

% ~ HG g = 0.9
config.scattering.amplitudeFunction{3}.mixtureMu = [0;1.764406405245083e+02;1.382801893975425;36.855826580175176;7.949781817711435;-0.078158767086068];
config.scattering.amplitudeFunction{3}.mixtureC = [-2.531024246969291;-1.731055330854194e+02;-2.831561238911424;-35.086689924296930;-7.714514400144169;-2.532042171876453];
config.scattering.amplitudeFunction{3}.mixtureAlpha = [0.105617258338313;0.067057292201984;0.758751296773679;0.206483373674102;0.359274660451266;0.326443781481564];

% ~ HG g = 0.8
config.scattering.amplitudeFunction{4}.mixtureMu = [0;8.567948260089990;40.747148947304666;1.512176451996613;-0.006999141602579];
config.scattering.amplitudeFunction{4}.mixtureC = [-2.531024246969291;-8.257797032110451;-38.877640140958086;-2.886694147327419;-2.531032411619821];
config.scattering.amplitudeFunction{4}.mixtureAlpha = [0.253871838456329;0.451553613239657;0.137923738827578;1.057954808434358;0.476692395924039];

% ~ HG g = 0.98
config.scattering.amplitudeFunction{5}.mixtureMu = [0;11.579090874816515;1.779309977673783;67.550452715965830;4.215886314000809e+02;2.882436622668382e+03;-0.005285363127619];
config.scattering.amplitudeFunction{5}.mixtureC = [-2.531024246969291;-10.967766980382587;-3.012069941353215;-65.175455016052640;-4.173824784349965e+02;-2.876308108469822e+03;-2.531028902808854];
config.scattering.amplitudeFunction{5}.mixtureAlpha = [0.063275130856032;0.164645985522835;0.319201646728633;0.107094124396163;0.065917372206027;0.026062432689170;0.164887237242181];

config.medium.type = 'Heterogeneous';
config.medium.xAxis = [-100,0,100];
config.medium.yAxis = [-100,100];
config.medium.zAxis = [0,10,20];
config.medium.materialGrid = [1,2,3,4];

% define the MFP
sigs = [0, 1/5,1/10,1/4,1/20];
config.medium.sigs = sigs;
config.medium.siga = sigs * 0;

[Xl,Zl] = ndgrid(focal_x,focal_z);
[Xv,Zv,tau_X] = ndgrid(focal_x,focal_z,tau_x);

% Illumination
config.illumination.type = 'GaussianBeam';
config.illumination.focalPoint.x = Xl(:);
config.illumination.focalPoint.y = 0 * Xl(:);
config.illumination.focalPoint.z = Zl(:);

config.illumination.focalDirection.x = zeros(size(Xl(:)));
config.illumination.focalDirection.y = zeros(size(Xl(:)));
config.illumination.focalDirection.z = ones(size(Xl(:)));

config.illumination.aperture = 0.4;

% View
config.view.type = 'GaussianBeam';
config.view.focalPoint.x = Xv(:) + tau_X(:);
config.view.focalPoint.y = 0 * Xv(:);
config.view.focalPoint.z = Zv(:);

config.view.focalDirection.x = zeros(size(Xv(:)));
config.view.focalDirection.y = zeros(size(Xv(:)));
config.view.focalDirection.z = ones(size(Xv(:)));

config.view.aperture = 0.4;
   

% Sampler
config.sampler.type = 'TemporalSampler';
config.sampler.t = 0;

config.sampler.D = zeros(1,5);

% Assuming no linear motion
config.sampler.U.x = zeros(1,5);
config.sampler.U.y = zeros(1,5);
config.sampler.U.z = zeros(1,5);

% Tracer
config.tracer.type = 'nee';

tic
[u, iter] = sstmc3D(config);
runTimeCorrelation = toc

u = u / sqrt(iter);

%%
u = reshape(u,numel(focal_x) * numel(focal_z),numel(focal_x) * numel(focal_z),numel(tau_x));
u_rendered = zeros(numel(focal_x) * numel(focal_z),numel(tau_x));

for ii = 1:1:numel(tau_x)
    u_rendered(:,ii) = diag(u(:,:,ii));
end

u_rendered = reshape(u_rendered,numel(focal_x),numel(focal_z),numel(tau_x));

maxVal = max(abs(u_rendered(:)));

figure
subplot(1,3,1)
imagesc(focal_z,focal_x,abs(u_rendered(:,:,1)),[0,maxVal])
title(['\tau_x = ',num2str(tau_x(1))])
xlabel('z')
ylabel('x')

subplot(1,3,2)
imagesc(focal_z,focal_x,abs(u_rendered(:,:,2)),[0,maxVal])
title(['\tau_x = ',num2str(tau_x(2))])
xlabel('z')
ylabel('x')

subplot(1,3,3)
imagesc(focal_z,focal_x,abs(u_rendered(:,:,3)),[0,maxVal])
title(['\tau_x = ',num2str(tau_x(3))])
xlabel('z')
ylabel('x')

colormap hot