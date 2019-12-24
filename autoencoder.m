clc;clear;close all;

% load the data provided by comsol
load('/data/outputnet_ksd10_numlay4_AcFunc_tansig20to1.mat');
load('/data/Datapoints.mat');
load('/data/COMSOL.mat');
load('/data/error_for_diff_topologies.mat'); 
YTrain=Y(:,1:3400);
YTest=Y(:,3401:end);
PseudoEncMSEvec=0;
PseudoEncNMSE=[];
PseudoEncSNMSE=[];
FullnetMSE=[];
FullnetNMSE=[];
%forming the pesudoencoder network for extracting underlying physics
for dd=10

numlay=[4,dd,10,20,20,30,30];   %topology of the autoencoder used for design domain
[m,ind]=min(numlay);    %finding the bottleneck

% set the size of the hidden layer for the autoencoder. For the autoencoder that you are going to train.
hiddenSize1 = numlay;


for i= 1:numel(numlay)
    DesignSpaceautoenc1.layers{i}.transferFcn =  'tansig';
end
DesignSpaceautoenc1.inputs{1}.processFcns = {};
DesignSpaceautoenc1.outputs{2}.processFcns = {};
DesignSpaceautoenc1 = feedforwardnet(hiddenSize1);
DesignSpaceautoenc1 = train(DesignSpaceautoenc1,YTrain,Xenc);
view(DesignSpaceautoenc1);

Xenc_Test=autoencHid1(XTest);

X_designnet=DesignSpaceautoenc1(YTest);
DesignMSE=mean(sum((Xenc_Test-X_designnet).^2));

PseudoEncMSEvec=[PseudoEncMSEvec, DesignMSE];
PseudoEncNMSE=[PseudoEncNMSE,DesignMSE/(sum(sum(Xenc_Test.^2))/length(Xenc_Test))];
PseudoEncSNMSE=[PseudoEncSNMSE,sqrt(DesignMSE/(sum(sum(Xenc_Test.^2))/length(Xenc_Test)))];

FullnetMSE=[FullnetMSE,mean(sum((autoencHid2(DesignSpaceautoenc1(YTest))-XTest).^2))];
FullnetNMSE=[FullnetNMSE,mean(sum((autoencHid2(DesignSpaceautoenc1(YTest))-XTest).^2))/((sum(sum(XTest.^2)))/length(XTest))];
FullnetSNMSE=sqrt(FullnetNMSE);
save(['PsedoEnc_' num2str(numlay) '_dd_' num2str(dd) '.mat'])
InputWeights=DesignSpaceautoenc1.IW{1};%design space to reduced design spaceconnections
U=DesignSpaceautoenc1.IW{1}';
figure(1)
bar(U,'BarWidth', 1)
ylim([-1 1])
legend({'DS Basis 1', 'DS Basis 2', 'DS Basis 3', 'DS Basis 4'},'FontSize',22,'FontName', 'Arial')
xticklabels({'h','n_1','n_2','n_3','w_1','w_2','w_3', 'p_1', 'p_2', 'p_3'})
set(gca,'fontsize',45,'FontName', 'Arial');
ylabel('Amplitude')
end
saveas(gcf, '../results/fig1.png')

%understanding part for 5-dimensional reduced design space
for dd=10

numlay=[5,dd,10,20,20,30,30];   %topology of the autoencoder used for design domain
[m,ind]=min(numlay);    %finding the bottleneck

hiddenSize1 = numlay;
for i= 1:numel(numlay)
    DesignSpaceautoenc1.layers{i}.transferFcn =  'tansig';
end
DesignSpaceautoenc1.inputs{1}.processFcns = {};
DesignSpaceautoenc1.outputs{2}.processFcns = {};
DesignSpaceautoenc1 = feedforwardnet(hiddenSize1);
DesignSpaceautoenc1 = train(DesignSpaceautoenc1,YTrain,Xenc);
view(DesignSpaceautoenc1);

Xenc_Test=autoencHid1(XTest);

X_designnet=DesignSpaceautoenc1(YTest);
DesignMSE=mean(sum((Xenc_Test-X_designnet).^2));

PseudoEncMSEvec=[PseudoEncMSEvec, DesignMSE];
PseudoEncNMSE=[PseudoEncNMSE,DesignMSE/(sum(sum(Xenc_Test.^2))/length(Xenc_Test))];
PseudoEncSNMSE=[PseudoEncSNMSE,sqrt(DesignMSE/(sum(sum(Xenc_Test.^2))/length(Xenc_Test)))];

FullnetMSE=[FullnetMSE,mean(sum((autoencHid2(DesignSpaceautoenc1(YTest))-XTest).^2))];
FullnetNMSE=[FullnetNMSE,mean(sum((autoencHid2(DesignSpaceautoenc1(YTest))-XTest).^2))/((sum(sum(XTest.^2)))/length(XTest))];
FullnetSNMSE=sqrt(FullnetNMSE);
save(['PsedoEnc_' num2str(numlay) '_dd_' num2str(dd) '.mat'])

InputWeights=DesignSpaceautoenc1.IW{1};
U=DesignSpaceautoenc1.IW{1}';
figure(2)
bar(U,'BarWidth', 1)
ylim([-1 1])
legend({'DS Basis 1', 'DS Basis 2', 'DS Basis 3', 'DS Basis 4','DS Basis 5'},'FontSize',22,'FontName', 'Arial')
xticklabels({'h','n_1','n_2','n_3','w_1','w_2','w_3', 'p_1', 'p_2', 'p_3'})
set(gca,'fontsize',45,'FontName', 'Arial');
ylabel('Amplitude')
end
saveas(gcf, '../results/fig2.png')

%forming the autoencoder platform for diffrent reduction rate
ksd=1;  
numlay=[25,20,ksd,20,25];   
[m,ind]=min(numlay);
hiddenSize1 = numlay;
for i= 1:numel(numlay)
autoenc1_1.layers{i}.transferFcn =  'tansig';
end
        
autoenc1_1.inputs{1}.processFcns = {};
        autoenc1_1.outputs{2}.processFcns = {};
        autoenc1_1 = feedforwardnet(hiddenSize1);
        autoenc1_1 = train(autoenc1_1,X,X);


        mn=1:20;
        rng=linspace(1,20,200);
        AE_Xintp_1=zeros(200,size(X,2));
    for i= 1:size(X,2)
        AE_Xintp_1(:,i)=interp1(mn,autoenc1_1(X(:,i)),rng);
    end
    
            autoencHid1_1=autoenc1_1;
            autoencHid2_1=autoenc1_1;
            autoencHid1_1.layerConnect(ind+1:end,:)=false;
            autoencHid1_1.outputConnect(1:end)=false;
            autoencHid1_1.outputConnect(ind)=true;
      

            autoencHid2_1.layerConnect(1:ind+1,:)=false;
            autoencHid2_1.inputConnect(1:end)=false;
            autoencHid2_1.inputConnect(ind+1)=true;
            autoencHid2_1.inputs{1}.size=numlay(1,ind);
            autoencHid2_1.IW{ind+1,1}=autoenc1_1.LW{ind+1,ind};
   
         ksd=5;  
        numlay=[25,20,ksd,20,25];   
        [m,ind]=min(numlay);
        ind=3;
        hiddenSize1 = numlay;

        for i= 1:numel(numlay)
            autoenc1_5.layers{i}.transferFcn =  'tansig';
        end
        
        autoenc1_5.inputs{1}.processFcns = {};
        autoenc1_5.outputs{2}.processFcns = {};
        autoenc1_5 = feedforwardnet(hiddenSize1);
        autoenc1_5 = train(autoenc1_5,X,X);
        mn=1:20;
        rng=linspace(1,20,200);


        AE_Xintp_5=zeros(200,size(X,2));
    for i= 1:size(X,2)
        AE_Xintp_5(:,i)=interp1(mn,autoenc1_5(X(:,i)),rng);
    end
    
            autoencHid1_5=autoenc1_5;
            autoencHid2_5=autoenc1_5;
            autoencHid1_5.layerConnect(ind+1:end,:)=false;
            autoencHid1_5.outputConnect(1:end)=false;
            autoencHid1_5.outputConnect(ind)=true;


            autoencHid2_5.layerConnect(1:ind+1,:)=false;
            autoencHid2_5.inputConnect(1:end)=false;
            autoencHid2_5.inputConnect(ind+1)=true;
            autoencHid2_5.inputs{1}.size=numlay(1,ind);
            autoencHid2_5.IW{ind+1,1}=autoenc1_5.LW{ind+1,ind};

             ksd=10;  
        numlay=[25,20,ksd,20,25];   
        [m,ind]=min(numlay);
        hiddenSize1 = numlay;

        for i= 1:numel(numlay)
            autoenc1_10.layers{i}.transferFcn =  'tansig';
        end
        
        autoenc1_10.inputs{1}.processFcns = {};
        autoenc1_10.outputs{2}.processFcns = {};
        autoenc1_10 = feedforwardnet(hiddenSize1);
        autoenc1_10 = train(autoenc1_10,X,X);



        AE_Xintp_10=zeros(200,size(X,2));
    for i= 1:size(X,2)
        AE_Xintp_10(:,i)=interp1(mn,autoenc1_10(X(:,i)),rng);
    end
    
    
                autoencHid1_10=autoenc1_10;
            autoencHid2_10=autoenc1_10;
            autoencHid1_10.layerConnect(ind+1:end,:)=false;
            autoencHid1_10.outputConnect(1:end)=false;
            autoencHid1_10.outputConnect(ind)=true;


            autoencHid2_10.layerConnect(1:ind+1,:)=false;
            autoencHid2_10.inputConnect(1:end)=false;
            autoencHid2_10.inputConnect(ind+1)=true;
            autoencHid2_10.inputs{1}.size=numlay(1,ind);
            autoencHid2_10.IW{ind+1,1}=autoenc1_10.LW{ind+1,ind};
 
         ksd=15;  
        numlay=[25,20,ksd,20,25];   
        [m,ind]=min(numlay);
        hiddenSize1 = numlay;

        for i= 1:numel(numlay)
            autoenc1_20.layers{i}.transferFcn =  'tansig';
        end
        
        autoenc1_20.inputs{1}.processFcns = {};
        autoenc1_20.outputs{2}.processFcns = {};
        autoenc1_20 = feedforwardnet(hiddenSize1);
        autoenc1_20 = train(autoenc1_20,X,X);



        AE_Xintp_20=zeros(200,size(X,2));
    for i= 1:size(X,2)
        AE_Xintp_20(:,i)=interp1(mn,autoenc1_20(X(:,i)),rng);
    end


                autoencHid1_20=autoenc1_20;
            autoencHid2_20=autoenc1_20;
            autoencHid1_20.layerConnect(ind+1:end,:)=false;
            autoencHid1_20.outputConnect(1:end)=false;
            autoencHid1_20.outputConnect(ind)=true;


            autoencHid2_20.layerConnect(1:ind+1,:)=false;
            autoencHid2_20.inputConnect(1:end)=false;
            autoencHid2_20.inputConnect(ind+1)=true;
            autoencHid2_20.inputs{1}.size=numlay(1,ind);
            autoencHid2_20.IW{ind+1,1}=autoenc1_20.LW{ind+1,ind};
            
%colors    
            col_1=[51,34,136];
            col_2=[170,68,153];
            col_3=[17,119,51];
            col_7=[136,20,238];
            col_4=[200,153,70];
            col_6=[204,102,119];
            col_5=[102,17,0];
            col_8=[68,170,153];
            col_9=[20,190,103];
            col_10=[90,110,253];
%sample for dimensionality reduction success           
fig_counter=5;

                       for smpnum=3401:4000
                       
                       O=interp1(mn,X(:,smpnum),rng,'spline');
                       A=interp1(mn,autoenc1_1(X(:,smpnum)),rng,'spline');
                       B=interp1(mn,autoenc1_5(X(:,smpnum)),rng,'spline');
                   C=interp1(mn,autoenc1_10(X(:,smpnum)),rng,'spline');
                       D=interp1(mn,autoenc1_20(X(:,smpnum)),rng,'spline');
                      figure(fig_counter)
                    hold on
                    plot(linspace(1250,1850,200),O,'color',col_1/255,'LineWidth',3)
                    plot(linspace(1250,1850,200),A,'--','color',col_2/255,'LineWidth',3)
                    plot(linspace(1250,1850,200),B,'-.','color',col_3/255,'LineWidth',3)
                    plot(linspace(1250,1850,200),C,':','color',col_4/255,'LineWidth',3)
                    plot(linspace(1250,1850,200),D,'color',col_5/255,'LineWidth',3)
                    xlabel('Wavelength (nm)')
                    ylabel('Reflectance')
                    axis([1250,1850,0,1])
                    set(gca,'fontsize',40,'FontName', 'Arial')
                    legend({'Original Data','1-Point Reconstruction','5-Point Reconstruction','10-Point Reconstruction','20-Point Reconstruction'},'FontSize',12)
                    hold off
                     saveas(gcf, ['../results/fig',num2str(fig_counter), '.png'])
                     fig_counter=fig_counter+1;
                     end

%error for diffrent topologies which is loaded from the data file
        figure(3)
        plot(log10(RS_rec_err(1,:)),'-s','color',col_1/255,'LineWidth',3','MarkerFaceColor',col_1/255)
        hold on
      
        xlabel('Reduced Response Space Dimensionality','LineWidth',3)
        ylabel('Reconstruction Logarithmic MSE ')
        set(gca,'fontsize',40,'FontName', 'Arial')
        grid minor
        hold off
        saveas(gcf, '../results/fig3.png')
        
figure(4)
        plot(RS_rec_err(1,:),'-s','color',col_1/255,'LineWidth',3','MarkerFaceColor',col_1/255)
        hold on
      
        xlabel('Reduced Response Space Dimensionality','LineWidth',3)
        ylabel('Reconstruction MSE ')
        set(gca,'fontsize',40,'FontName', 'Arial')
        grid minor
   
        hold off
        saveas(gcf, '../results/fig4.png')