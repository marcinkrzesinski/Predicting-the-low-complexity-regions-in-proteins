%% Predicting the low complexity regions in proteins using machine learning

% Objective: 
% To create a predictive machine learning model capable of 
% classifying given regions of a protein's amino acid sequence (regions of low or no complexity).

% Assumptions: 
% The project will be carried out using technologies such as 
% Matlab, Deep Learning Toolbox and Bioinformatics Toolbox

% Motivation: 
% To create an alternative form of prediction of LCR regions.

% Description of topic: 
% Protein low complexity regions are fragments 
% of an amino acid sequence that are characterized by tandem or multiple repeats of the same amino acid. 
% While this was once thought to be of little importance, today's knowledge 
% and research confirms links of these repeats to e.g. diseases such as Huntington's chorea.

% Expected results: 
% A machine learning model program capable of accurate
% (relative to legacy programs such as SEG) prediction of protein regions 
% of low complexity.

%%
% Read dataset and target (lcr)
db = fastaread('db-test-2.fasta'); %db-test-2.fasta
lcr = fastaread('lcr-test-2.fasta'); %lcr-test-2.fasta

% Number of elements in databse (N_db) and taregt (N_lcr)
N_db = numel(db);
N_lcr = numel(lcr);


% == aa2int for db
for i = 1:N_db
	seq_int = aa2int(db(i).Sequence);
	db(i).SequenceInt = seq_int;
end
% == aa2int for lcr
for i = 1:N_lcr
	seq_int = aa2int(lcr(i).Sequence);
	lcr(i).SequenceInt = seq_int;
end

id = db(1).Header;            % annotation of a given protein sequence 
seq = db(1).Sequence;         % protein sequence
str = db(1).SequenceInt;      % Sequence using Integer

% Add columns: SeqLCR and SeqLCRInt to database [db]
for i = 1:N_db
    db(i).SeqLCR = lcr(i).Sequence;
    db(i).SeqLCRInt = lcr(i).SequenceInt;
end

%% Binarization of the inputs and targets 
tem2 = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'];

% Input
for i = 1:N_db
    seq = db(i).Sequence;
    N_seq = numel(seq);
    inputA = zeros(20,N_seq);
   
    for j = 1:N_seq
        if(seq(j) == 'A'); inputA(1,j)=1; end
        if(seq(j) == 'C'); inputA(2,j)=1; end
        if(seq(j) == 'D'); inputA(3,j)=1; end
        if(seq(j) == 'E'); inputA(4,j)=1; end
        if(seq(j) == 'F'); inputA(5,j)=1; end
        if(seq(j) == 'G'); inputA(6,j)=1; end
        if(seq(j) == 'H'); inputA(7,j)=1; end
        if(seq(j) == 'I'); inputA(8,j)=1; end
        if(seq(j) == 'K'); inputA(9,j)=1; end
        if(seq(j) == 'L'); inputA(10,j)=1; end
        if(seq(j) == 'M'); inputA(11,j)=1; end
        if(seq(j) == 'N'); inputA(12,j)=1; end
        if(seq(j) == 'P'); inputA(13,j)=1; end
        if(seq(j) == 'Q'); inputA(14,j)=1; end
        if(seq(j) == 'R'); inputA(15,j)=1; end
        if(seq(j) == 'S'); inputA(16,j)=1; end
        if(seq(j) == 'T'); inputA(17,j)=1; end
        if(seq(j) == 'V'); inputA(18,j)=1; end
        if(seq(j) == 'W'); inputA(19,j)=1; end
        if(seq(j) == 'Y'); inputA(20,j)=1; end 
    end
    db(i).SeqMatrix = inputA;
    
    % Short Seq Protein
    win = 5; % sliding windows
    startLCR = strfind(db(i).SequenceInt, db(i).SeqLCRInt);
    endLCR = startLCR + numel(db(i).SeqLCRInt); 
    lengthLCR = endLCR - startLCR;
    
    if (startLCR < win+1)
        startLCR = 1;
    elseif (endLCR > N_seq-win)
        difff = N_seq - endLCR;
        endLCR = endLCR + difff;
        startLCR = startLCR - win;
    else
        startLCR = startLCR - win;
        endLCR = endLCR + win;
    end
    
    colsLCR = lengthLCR + (win*2)+1;
    
    % Short Seq Protein (INT)
    shortSeqProtein = zeros(20,colsLCR);
    db(i).shortSeqProtein = db(i).SeqMatrix(:,startLCR:endLCR);
    
    % Short Seq Protein (AA) -> find [start,stop] for shortSeqLCR
    db(i).shortSeqP = int2aa(db(i).SequenceInt(:,startLCR:endLCR));
    n_shortSeqP = numel(db(i).shortSeqP);
    
    
    % -------------------------------------------------------
    % Short LCR
    % Find [start,stop]
    startShortLCR = strfind(db(i).shortSeqP, db(i).SeqLCR); % start
    endShortLCR = startShortLCR + lengthLCR - 1; %end
    
    inputShortSeqLCR = db(i).shortSeqProtein(:,startShortLCR:endShortLCR);
    
    % shortLCR + padding
    AToStart = zeros(20,startShortLCR-1);
    diffLCR = n_shortSeqP - endShortLCR;
    EndToEnd = zeros(20,diffLCR);
    db(i).shortSeqLCR = [AToStart inputShortSeqLCR EndToEnd];
     
end


% Target
for i = 1:N_db
    seq = db(i).SeqLCR;
    N_seq = numel(seq);
    
    N_seq_db = numel(db(i).Sequence);
    inputT = zeros(20,N_seq);
    
    % start | end for LCR
    startLCR = strfind(db(i).SequenceInt, db(i).SeqLCRInt);
    endLCR = startLCR + numel(db(i).SeqLCRInt);
    
    aToStart = zeros(20,startLCR-1);
    diff = N_seq_db - endLCR + 1;
    EndtoEnd = zeros(20,diff);
    
    for j = 1:N_seq
        if(seq(j) == 'A'); inputT(1,j)=1; end
        if(seq(j) == 'C'); inputT(2,j)=1; end
        if(seq(j) == 'D'); inputT(3,j)=1; end
        if(seq(j) == 'E'); inputT(4,j)=1; end
        if(seq(j) == 'F'); inputT(5,j)=1; end
        if(seq(j) == 'G'); inputT(6,j)=1; end
        if(seq(j) == 'H'); inputT(7,j)=1; end
        if(seq(j) == 'I'); inputT(8,j)=1; end
        if(seq(j) == 'K'); inputT(9,j)=1; end
        if(seq(j) == 'L'); inputT(10,j)=1; end
        if(seq(j) == 'M'); inputT(11,j)=1; end
        if(seq(j) == 'N'); inputT(12,j)=1; end
        if(seq(j) == 'P'); inputT(13,j)=1; end
        if(seq(j) == 'Q'); inputT(14,j)=1; end
        if(seq(j) == 'R'); inputT(15,j)=1; end
        if(seq(j) == 'S'); inputT(16,j)=1; end
        if(seq(j) == 'T'); inputT(17,j)=1; end
        if(seq(j) == 'V'); inputT(18,j)=1; end
        if(seq(j) == 'W'); inputT(19,j)=1; end
        if(seq(j) == 'Y'); inputT(20,j)=1; end
    end
    db(i).SeqLCRMatrix = [aToStart inputT EndtoEnd];    
end


% Concatenate inputs & targets
SeqMatrixVertically = double([db.SeqMatrix]);
SeqLCRMatrixVertically = double([db.SeqLCRMatrix]);

myShortInput = double([db.shortSeqProtein]);
myShortTarget = double([db.shortSeqLCR]);


%% Network design
hsize = 1000; % Hidden Layer - neurons

net = patternnet(hsize); % Pattern Recognition Neural Network (hidden layer)

% One Hidden Layer
net.layers{1} % hidden layer
net.layers{2} % output layer 

% Transfer Function
net.layers{1}.transferFcn = 'logsig';
% net.layers{2}.transferFcn = 'logsig';
% net.layers{3}.transferFcn = 'logsig';

% Train the network
[net,tr] = train(net,myShortInput,myShortTarget);
view(net);

% Parameters
hsize = [3 4 2];
net3 = patternnet(hsize);

hsize = 20;
net20 = patternnet(hsize);

% === assign random values in the range -.1 and .1 to the weights 
net20.IW{1} = -.1 + (.1 + .1) .* rand(size(net20.IW{1}));
net20.LW{2} = -.1 + (.1 + .1) .* rand(size(net20.LW{2}));

net20 = train(net20,myShortInput,myShortTarget);

O20 = sim(net20,myShortInput);
numWeightsAndBiasesNet = length(getx(net)); % Number of Weights and Biases for net
numWeightsAndBiasesNet20 = length(getx(net20)); % Number of Weights and Biases for net20

%% Rest defaults [net]:
% net.trainParam.epochs: 1000
% net.trainParam.goal: 0
% net.trainParam.min_grad: 1e-06;
% net.trainParam.max_fail: 6
% net.trainParam.sgima: 5e-05
% net.trainParam.lambda: 5e-07
% 
% name: 'Pattern Recognition Neural Network'
% 
% net.performParam: .regularization, .normalization
% net.performFcn: 'crossentropy'
% net.trainFCN: 'trainscg'

% IW: {2x1 cell} containing 1 input weight matrix
% LW: {2x2 cell} containing 1 layer weight matrix
% b: {2x1 cell} containing 2 bias vectors

%% Structural assignments

% TRAIN DATA SET
[i,j] = find(SeqLCRMatrixVertically(:,tr.trainInd));
Atrain = sum(i == 1)/length(i);
Ctrain = sum(i == 2)/length(i);
Dtrain = sum(i == 3)/length(i);
Etrain = sum(i == 4)/length(i);
Ftrain = sum(i == 5)/length(i);
Gtrain = sum(i == 6)/length(i);
Htrain = sum(i == 7)/length(i);
Itrain = sum(i == 8)/length(i);
Ktrain = sum(i == 9)/length(i);
Ltrain = sum(i == 10)/length(i);
Mtrain = sum(i == 11)/length(i);
Ntrain = sum(i == 12)/length(i);
Ptrain = sum(i == 13)/length(i);
Qtrain = sum(i == 14)/length(i);
Rtrain = sum(i == 15)/length(i);
Strain = sum(i == 16)/length(i);
Ttrain = sum(i == 17)/length(i);
Vtrain = sum(i == 18)/length(i);
Wtrain = sum(i == 19)/length(i);
Ytrain = sum(i == 20)/length(i);

figure()
pie([Atrain;Ctrain;Dtrain;Etrain;Ftrain;Gtrain;Htrain;Itrain;Ktrain;Ltrain;Mtrain;Ntrain;Ptrain;Qtrain;Rtrain;Strain;Ttrain;Vtrain;Wtrain;Ytrain]);
title('Structural assignments in TRAINING data set');
legend('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y');

% % =========================================================
% % VALIDATION DATA SET
[i,j] = find(SeqLCRMatrixVertically(:,tr.valInd));
Atrain = sum(i == 1)/length(i);
Ctrain = sum(i == 2)/length(i);
Dtrain = sum(i == 3)/length(i);
Etrain = sum(i == 4)/length(i);
Ftrain = sum(i == 5)/length(i);
Gtrain = sum(i == 6)/length(i);
Htrain = sum(i == 7)/length(i);
Itrain = sum(i == 8)/length(i);
Ktrain = sum(i == 9)/length(i);
Ltrain = sum(i == 10)/length(i);
Mtrain = sum(i == 11)/length(i);
Ntrain = sum(i == 12)/length(i);
Ptrain = sum(i == 13)/length(i);
Qtrain = sum(i == 14)/length(i);
Rtrain = sum(i == 15)/length(i);
Strain = sum(i == 16)/length(i);
Ttrain = sum(i == 17)/length(i);
Vtrain = sum(i == 18)/length(i);
Wtrain = sum(i == 19)/length(i);
Ytrain = sum(i == 20)/length(i);

figure()
pie([Atrain;Ctrain;Dtrain;Etrain;Ftrain;Gtrain;Htrain;Itrain;Ktrain;Ltrain;Mtrain;Ntrain;Ptrain;Qtrain;Rtrain;Strain;Ttrain;Vtrain;Wtrain;Ytrain]);
title('Structural assignments in VALIDATION data set');
legend('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y');

% % =========================================================
% % TESTING DATA SET
[i,j] = find(SeqLCRMatrixVertically(:,tr.testInd));
Atrain = sum(i == 1)/length(i);
Ctrain = sum(i == 2)/length(i);
Dtrain = sum(i == 3)/length(i);
Etrain = sum(i == 4)/length(i);
Ftrain = sum(i == 5)/length(i);
Gtrain = sum(i == 6)/length(i);
Htrain = sum(i == 7)/length(i);
Itrain = sum(i == 8)/length(i);
Ktrain = sum(i == 9)/length(i);
Ltrain = sum(i == 10)/length(i);
Mtrain = sum(i == 11)/length(i);
Ntrain = sum(i == 12)/length(i);
Ptrain = sum(i == 13)/length(i);
Qtrain = sum(i == 14)/length(i);
Rtrain = sum(i == 15)/length(i);
Strain = sum(i == 16)/length(i);
Ttrain = sum(i == 17)/length(i);
Vtrain = sum(i == 18)/length(i);
Wtrain = sum(i == 19)/length(i);
Ytrain = sum(i == 20)/length(i);

figure()
pie([Atrain;Ctrain;Dtrain;Etrain;Ftrain;Gtrain;Htrain;Itrain;Ktrain;Ltrain;Mtrain;Ntrain;Ptrain;Qtrain;Rtrain;Strain;Ttrain;Vtrain;Wtrain;Ytrain]);
title('Structural assignments in TESTING data set');
legend('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y');


%% Performance plot
figure()
plotperform(tr)

%% Train param and train state

% Display training parameters
net.trainParam

% Plot validation checks and gradient
figure()
plottrainstate(tr)

%% Analyzing the Network Response - Plot Confusion and ROC

% Output - O
O = sim(net,myShortInput);

% Confusion
fh = figure();
fh.WindowState = 'maximized'; % full screen plotconfusion
plotconfusion(myShortTarget,O);

% ROC - Receiver Operating Characteristic
% true positive rate (sensitivity) and false positive rate (specificity)
figure()
plotroc(myShortTarget,O);

%% Refining the Neural Network for More Accurate Results
hsize = [3 4 2];
net3 = patternnet(hsize);

hsize = 20;
net20 = patternnet(hsize);

% === assign random values in the range -.1 and .1 to the weights 
% net3.IW{1} = -.1 + (.1 + .1) .* rand(size(net20.IW{1}));
% net3.LW{2} = -.1 + (.1 + .1) .* rand(size(net20.LW{2}));

net3 = train(net20,myShortInput,myShortTarget);

%% NET 20

net20 = train(net20,myShortInput,myShortTarget);

O20 = sim(net20,myShortInput);
numWeightsAndBiasesNet = length(getx(net)); % Number of Weights and Biases for net
numWeightsAndBiasesNet20 = length(getx(net20)); % Number of Weights and Biases for net20


%% Assessing Network Performance

% Target
[i,j] = find(myShortTarget);
Atrain = sum(i == 1);
Ctrain = sum(i == 2);
Dtrain = sum(i == 3);
Etrain = sum(i == 4);
Ftrain = sum(i == 5);
Gtrain = sum(i == 6);
Htrain = sum(i == 7);
Itrain = sum(i == 8);
Ktrain = sum(i == 9);
Ltrain = sum(i == 10);
Mtrain = sum(i == 11);
Ntrain = sum(i == 12);
Ptrain = sum(i == 13);
Qtrain = sum(i == 14);
Rtrain = sum(i == 15);
Strain = sum(i == 16);
Ttrain = sum(i == 17);
Vtrain = sum(i == 18);
Wtrain = sum(i == 19);
Ytrain = sum(i == 20);

% Output
competO = compet(O);
[u,v] = find(competO);
AtrainO = sum(u == 1);
CtrainO = sum(u == 2);
DtrainO = sum(u == 3);
EtrainO = sum(u == 4);
FtrainO = sum(u == 5);
GtrainO = sum(u == 6);
HtrainO = sum(u == 7);
ItrainO = sum(u == 8);
KtrainO = sum(u == 9);
LtrainO = sum(u == 10);
MtrainO = sum(u == 11);
NtrainO = sum(u == 12);
PtrainO = sum(u == 13);
QtrainO = sum(u == 14);
RtrainO = sum(u == 15);
StrainO = sum(u == 16);
TtrainO = sum(u == 17);
VtrainO = sum(u == 18);
WtrainO = sum(u == 19);
YtrainO = sum(u == 20);


competOfind = find(competO);

% compute fraction of correct predictions when a given state is observed
pcObs(1) = sum(Atrain & AtrainO)/sum (Atrain); % state A
pcObs(2) = sum(Ctrain & CtrainO)/sum (Ctrain); % state C
pcObs(3) = sum(Dtrain & DtrainO)/sum (Dtrain); % state D
pcObs(4) = sum(Etrain & EtrainO)/sum (Etrain); % state E
pcObs(5) = sum(Ftrain & FtrainO)/sum (Ftrain); % state F
pcObs(6) = sum(Gtrain & GtrainO)/sum (Gtrain); % state G
pcObs(7) = sum(Htrain & HtrainO)/sum (Htrain); % state H
pcObs(8) = sum(Itrain & ItrainO)/sum (Itrain); % state I
pcObs(9) = sum(Ktrain & KtrainO)/sum (Ktrain); % state K
pcObs(10) = sum(Ltrain & LtrainO)/sum (Ltrain); % state L
pcObs(11) = sum(Mtrain & MtrainO)/sum (Mtrain); % state M
pcObs(12) = sum(Ntrain & NtrainO)/sum (Ntrain); % state N
pcObs(13) = sum(Ptrain & PtrainO)/sum (Ptrain); % state P
pcObs(14) = sum(Qtrain & QtrainO)/sum (Qtrain); % state Q
pcObs(15) = sum(Rtrain & RtrainO)/sum (Rtrain); % state R
pcObs(16) = sum(Strain & StrainO)/sum (Strain); % state S
pcObs(17) = sum(Ttrain & TtrainO)/sum (Ttrain); % state T
pcObs(18) = sum(Vtrain & VtrainO)/sum (Vtrain); % state V
pcObs(19) = sum(Wtrain & WtrainO)/sum (Wtrain); % state W
pcObs(20) = sum(Ytrain & YtrainO)/sum (Ytrain); % state Y

% compute fraction of correct predictions when a given state is predicted
pcPred(1) = sum(Atrain & AtrainO)/sum (AtrainO); % state A
pcPred(2) = sum(Ctrain & CtrainO)/sum (CtrainO); % state C
pcPred(3) = sum(Dtrain & DtrainO)/sum (DtrainO); % state D
pcPred(4) = sum(Etrain & EtrainO)/sum (EtrainO); % state E
pcPred(5) = sum(Ftrain & FtrainO)/sum (FtrainO); % state F
pcPred(6) = sum(Gtrain & GtrainO)/sum (GtrainO); % state G
pcPred(7) = sum(Htrain & HtrainO)/sum (HtrainO); % state H
pcPred(8) = sum(Itrain & ItrainO)/sum (ItrainO); % state I
pcPred(9) = sum(Ktrain & KtrainO)/sum (KtrainO); % state K
pcPred(10) = sum(Ltrain & LtrainO)/sum (LtrainO); % state L
pcPred(11) = sum(Mtrain & MtrainO)/sum (MtrainO); % state M
pcPred(12) = sum(Ntrain & NtrainO)/sum (NtrainO); % state N
pcPred(13) = sum(Ptrain & PtrainO)/sum (PtrainO); % state P
pcPred(14) = sum(Qtrain & QtrainO)/sum (QtrainO); % state Q
pcPred(15) = sum(Rtrain & RtrainO)/sum (RtrainO); % state R
pcPred(16) = sum(Strain & StrainO)/sum (StrainO); % state S
pcPred(17) = sum(Ttrain & TtrainO)/sum (TtrainO); % state T
pcPred(18) = sum(Vtrain & VtrainO)/sum (VtrainO); % state V
pcPred(19) = sum(Wtrain & WtrainO)/sum (WtrainO); % state W
pcPred(20) = sum(Ytrain & YtrainO)/sum (YtrainO); % state Y


% compare quality indices of prediction
fh = figure();
fh.WindowState = 'maximized';
bar([pcObs' pcPred'] * 100);
ylabel('Correctly predicted positions (%)');
xlabel('Amino acids');
x = {'A';'C';'D';'E';'F';'G';'H';'I';'K';'L';'M';'N';'P';'Q';'R';'S';'T';'V';'W';'Y'};
legend({'Observed','Predicted'});
set(gca,'XTickLabel',x, 'XTick', 1:numel(x));


%% TESTS
basecount(seq)

% = Amino acid count chart -> bar + pie
seqLCR = db(1).SeqLCR;

figure
aacount(seq, 'chart', 'bar'); %count amino in seq
figure
aacount(seq,'chart','pie');
figure
aacount(seqLCR,'chart','pie');


% = Compare two Seqs -> dotplot
seq = db(1).Sequence;  
seqLCR = db(1).SeqLCR;

seqdotplot(seq,seqLCR,4,3);
ylabel('Seq DB')
xlabel('seq LCR')
uif = gcf;
uif.Position(:) = [100 100 1280 800]; % Resize the figure.



%% Conclusion
% The goal of this study was to develop a predictive learning model capable 
% of classifying low complexity regions from amino acid sequences 
% of proteins, which was accomplished. The resulting
% model is capable of determining with an accuracy of 75.5%
% the amino acid sequences of regions of low complexity from protein sequences
% compared to the results obtained by the SEG method, as shown in the figures in the
% results section.

