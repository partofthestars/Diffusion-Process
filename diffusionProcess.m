%%########################################################################
%%
%%	PPGI: Diffusion Procss
%%
%%	Christian S. Pilz, Jarek Krajewski, Vladimir Blazek.
%%      On the Diffusion Process for Heart Rate Estimation from Face Videos under Realistic Conditions. 
%%      Pattern Recognition: 39th German Conference, GCPR 2017, Basel, Switzerland. 
%%      Proceedings (Lecture Notes in Computer Science), Springer, 2017
%%
%%      Author   : Christian S. Pilz
%%      Date     : 15.03.2017
%%
%%      Contact  : cpi@partofthestars.com
%%      Web Page : www.partofthestars.com
%%
%%      Version  : Alpha RA 1.0
%%
%%      License  : GPL v3
%%
%%########################################################################
%%
%%	diffusionProcess.m:
%%

clear all; close all;

base_system_dir='./data/images/';

load('./data/CMS50E_ppg.mat');

% analysis loop
%

frames=1452;

for f=1:frames
    
    file=[base_system_dir num2str(f+24) '.png'];
    
    I = imread(file);
    [rows cols dim]=size(I);
    
    %find all skin pixels
    [out bin]=generate_skinmap(file);
    
    r=double(I(:,:,1));
    r=r(bin==1);
    
    g=double(I(:,:,2));
    g=g(bin==1);
    
    b=double(I(:,:,3));
    b=b(bin==1);

    values=[r g b];
    
    %raw mean traces
    trace(f,:)=mean(values);
    
    % spatial subspace rotation features
    %
    
    %spatial RGB correlation:
    C=(values'*values)/(rows*cols);
    [V,D] = eig(C);

    [D,I] = sort(diag(D),'descend');
     V = V(:, I);

    U{f}=V;
    Sigmas{f}=D;
 
    if f>1
        %rotation between the skin vector and orthonormal plane
        R{f-1}=[U{f}(:,1)'*U{f-1}(:,2) U{f}(:,1)'*U{f-1}(:,3)];
         
        %scale change
        S{f-1}=[sqrt(Sigmas{f}(1)/Sigmas{f-1}(2)) sqrt(Sigmas{f}(1)/Sigmas{f-1}(3)) ];
        SR{f-1}=S{f-1}.*R{f-1};
        SR_backprojected{f-1}=SR{f-1}*[U{f-1}(:,2)'; U{f-1}(:,3)'];
        
        signal_u(f-1,:)=U{f}(:,1)';
        signal_r(f-1,:)=R{f-1};
        signal_s(f-1,:)=S{f-1};
        signal_sr(f-1,:)=SR{f-1};
        signal_sr_b(f-1,:)=SR_backprojected{f-1};
    end
    
end

 sigma=std(signal_sr_b(:,1))/std(signal_sr_b(:,2));  
 p=signal_sr_b(:,2);%-sigma*signal_sr_b(:,3);
 pp=signal_sr_b(:,1)-sigma*signal_sr_b(:,2);
 pp=pp-mean(pp);
 
 fs=25;
 windowSize=3;%sec
 overLap=2;%sec
 
 [blocks_one]=buffer(signal_sr_b(:,1),windowSize*fs,windowSize*fs-1)';
 [blocks_two]=buffer(signal_sr_b(:,2),windowSize*fs,windowSize*fs-1)';
 [blocks_three]=buffer(signal_sr_b(:,3),windowSize*fs,windowSize*fs-1)';
 
 [frames dim]=size(blocks_one);

for i=1:frames
    sigma=std(blocks_one(i,:))/std(blocks_two(i,:));  
	
    p_block=blocks_one(i,:)-sigma*blocks_two(i,:);
    pulse(i,:)=double(p_block-mean(p_block));
end


%low-pass filter
%
cutoff=2.0;%hz
fNorm = cutoff / (fs/2);                    
[b,a] = butter(10, fNorm, 'low');        
                                        
p_blocked = filtfilt(b, a, double(pulse(:,end)));

%diffusion process
%

%estimate of measurement noise standard deviation
%in the IMM model.
sd = 0.025;
%the process spectral density for the bias model rep-
%resents the continuous time noise in the sensor signal
bq = 0.001;
%the resonator process noise spectral density defines
%the continuous-time variation of the resonator sig-
%nals. adjust primarily this parameter to control the
%behavior of the periodic signals.
qr = 0.0001;

%number of harmonics including fundamental
nharm=1;
%time delta
dt=1/fs;
%transition probability between consecutive steps of
%frequencies (i.e. the probability of a jump from e.g. 70 bpm to 71 bpm)
ptrans = 0.15;
%transition probability between all steps of frequencies
%(i.e. the probability of a jump from e.g. 70 bpm to 80 bpm).
poverall = 0;
%frequency search space
freqlist=45:120;

p_blocked_dp=tracker(p_blocked,dt,freqlist,nharm,bq,sd,qr,ptrans,poverall);


%raw signal
raw=trace(:,2);
raw = filtfilt(b, a, double(raw(:,end)));
raw = diff(raw);

%ica: to see that it's more or less rubbish
%

%filter
% trace(:,1) = filtfilt(b, a, double(trace(:,1)));
% trace(:,2) = filtfilt(b, a, double(trace(:,2)));
% trace(:,3) = filtfilt(b, a, double(trace(:,3)));
% %detrend
% raw_rgb(:,1) = diff(trace(:,1));
% raw_rgb(:,2) = diff(trace(:,2));
% raw_rgb(:,3) = diff(trace(:,3));
% %source sepration
% B = jadeR(raw_rgb',3);
% s = B' * raw_rgb';
% figure
% subplot(3,1,1);
% spectrogram(s(1,:),128,120,f,fs,'yaxis');
% ylim([0 3]);
% subplot(3,1,2);
% spectrogram(s(2,:),128,120,f,fs,'yaxis');
% ylim([0 3]);
% subplot(3,1,3);
% spectrogram(s(3,:),128,120,f,fs,'yaxis');
% ylim([0 3]);


%pulse oximeter
%

fs=60;

cutoff=2.0;%hz
fNorm = cutoff / (fs/2);                    
[b,a] = butter(10, fNorm, 'low');
ppg = filtfilt(b, a, double(ppg(:,end)));
cutoff=0.5;%hz
fNorm = cutoff / (fs/2);
[b,a] = butter(10, fNorm, 'high');                                      
ppg = filtfilt(b, a, double(ppg(:,end)));
ppg = diff(ppg);

%plot results
%

figure;

fs=25;

y=raw';
L=length(y);
NFFT = 2^nextpow2(L);
f = fs/2*linspace(0,1,NFFT/2+1);

subplot(4,2,1);
plot(raw);
title('Derivatives of green-channel means');
xlabel('Frames');
subplot(4,2,2);
spectrogram(raw,128,120,f,fs,'yaxis');
title('Spectrogram')
ylim([0 3]);

subplot(4,2,3);
plot(p_blocked);
title('Spatial Subspace Rotation')
xlabel('Frames');

y=p_blocked';
L=length(y);
NFFT = 2^nextpow2(L);
f = fs/2*linspace(0,1,NFFT/2+1);

subplot(4,2,4);
spectrogram(p_blocked,128,120,f,fs,'yaxis');
title('Spectrogram')
ylim([0 3]);

subplot(4,2,5);
plot(p_blocked_dp);
title('Diffusion Process')
xlabel('Frames');

y=p_blocked_dp;
L=length(y);
NFFT = 2^nextpow2(L);
f = fs/2*linspace(0,1,NFFT/2+1);

subplot(4,2,6);
spectrogram(p_blocked_dp,128,120,f,fs,'yaxis');
title('Spectrogram')
ylim([0 3]);

subplot(4,2,7);
plot(ppg);
title('CMS50E Pulse Oximeter')
xlabel('Frames');

fs=60;
y=ppg;
L=length(y);
NFFT = 2^nextpow2(L);
f = fs/2*linspace(0,1,NFFT/2+1);

subplot(4,2,8);
spectrogram(ppg,128,120,f,fs,'yaxis');
title('Spectrogram')
ylim([0 3]);
