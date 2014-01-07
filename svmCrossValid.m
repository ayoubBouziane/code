%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is a cross validation wrapper for libSVM using precomputed
% kernel. Given a C array and a fold number, it finds out the best C value.
% If assigned, corresponding SVM model can also be generated (fNeedModel). 
% This function is designed to find one C for one model. If only one C is 
% needed for multiple models, then use AccList output parameter to find out
% the best C for multiple models. Gamma is heuristic set in the function.
% 
% Input :
%      libSVMpath : the local libSVM path.
%      trainFtr : a MxN training feature matrix. M is the number of
%      instances while N is the feature dimension.
%      trainGroup : the positive and negative label of the training data.
%      It should be a MxR matrix. M is the number of instances and R is the 
%      number of classifiers. Each instance should be labeled individually.
%      ClassLabel : the class label of the training data. It is a Mx1
%      matrix, labeling the class of each training instance.
%      C : the C array to cross validate on.
%      numOfFold : the number of fold to cross validate.
%      fNeedModel : logical flag. 1 means svmModel output needed, 0
%      otherwise.
%      fUniqueC : logical flag. 1 means that use 1 c for all the classifiers
%      and 0 means that use different c for different classifiers.
%      fPrecomputed : logical flag. Valid only when fNeedModel is set as 1.
%      1 means the output svmModels are trained using precomputed kernel, 0
%      means using the normal way to train the models.
% Output :
%      CBest : the final choice of C. When fUniqueC is 1, it is a value, a
%      vector other wise.
%      svmModels : the SVM models trained using CBest on the whole training
%      data (trainFtr).     
%      gammaOut : when pre-computed kernel model is chosen, the
%      corresponding gamma.
% Author : Yi Li, Computer Vision, QMUL
%
% Version : 1.0 Dec 9, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [CBest, svmModels, gammaOut] = svmCrossValid(libSVMpath, trainFtr, trainGroup, ClassLabel, C, numOfFold, fNeedModel,fUniqueC, fPrecomputed)   
    %% pre-settings
    addpath(libSVMpath);
    numClassifier = size(trainGroup,2);
    szC = size(C,2);
    accList = zeros(numClassifier,szC); 
    svmModels = [];
    gammaOut = [];
    %% sorting the data
    [~,ind] = sort(ClassLabel);
    trainFtr = trainFtr(ind,:);
    trainGroup = trainGroup(ind,:);
    
    %% organize the training data by class and check if every class has sufficient data for the numOfFold
    classes = unique(ClassLabel);
    numClass = size(classes,1);
    
    % sort training data into classes
    trainClass = cell(numClass,1);
    groupClass = cell(numClass,1);
    
    % size container
    classSizes = zeros(numClass,1);
    foldSizes = zeros(numClass,1);
    
    for i = 1 : numClass
        classSizes(i) = sum(ClassLabel == classes(i));
        if classSizes(i) < numOfFold
            disp('Some class does not have enough training data.');
            return;
        end
        
        trainClass{i} = trainFtr(ClassLabel==classes(i),:);
        groupClass{i} = trainGroup(ClassLabel==classes(i),:);
        foldSizes(i) = floor(classSizes(i)/numOfFold);
    end
    
    %%  divide training data to different folds
    foldData = cell(numOfFold,1);
    foldGroup = cell(numOfFold,1);
    
    for i = 1 : numOfFold
        counter = 0;
        for j = 1 : numClass
            if i ~= numOfFold
                foldData{i}(counter+1:counter+foldSizes(j),:) = trainClass{j}((i-1)*foldSizes(j)+1:(i-1)*foldSizes(j)+foldSizes(j),:);
                foldGroup{i}(counter+1:counter+foldSizes(j),:) = groupClass{j}((i-1)*foldSizes(j)+1:(i-1)*foldSizes(j)+foldSizes(j),:);
                counter = counter + foldSizes(j);
            else
                lastFoldSize = classSizes(j) - (numOfFold-1) * foldSizes(j); 
                foldData{i}(counter+1:counter+lastFoldSize,:) = trainClass{j}((i-1)*foldSizes(j)+1:(i-1)*foldSizes(j)+lastFoldSize,:);
                foldGroup{i}(counter+1:counter+lastFoldSize,:) = groupClass{j}((i-1)*foldSizes(j)+1:(i-1)*foldSizes(j)+lastFoldSize,:);
                counter = counter + lastFoldSize;
            end
        end
    end
    
    %% cross validation
    %%% c loop %%%
    for c = 1 : szC
        fprintf('C round: %d/%d\n',c,szC);

        %%% fold loops %%%
        for fold_num = 1 : numOfFold
            fprintf('fold round: %d/%d\n',fold_num,numOfFold);          

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%% 1. train the kernel matrix%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % for training set
            Training = [];
            TrainingGroup = [];
            counter = 1;
            for i = 1 : numOfFold
                if i ~= fold_num
                    foldSize = size(foldData{i},1);
                    Training(counter:counter+foldSize-1,:) = foldData{i};
                    TrainingGroup(counter:counter+foldSize-1,:) = foldGroup{i};
                    counter = counter + foldSize;
                end
            end
            
            distMx = pdist2(Training,Training).^2;
            gamma = 1/mean(mean(distMx));
            trainKernel = exp(-gamma*distMx);
            trainLabel = (1:1:size(trainKernel,1));

            % for test set
            Sample = foldData{fold_num};
            SampleGroup = foldGroup{fold_num};
            distMx = pdist2(Sample,Training).^2;
            sampleKernel = exp(-gamma*distMx);
            sampleLabel = (1:1:size(sampleKernel,1));
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%% 2. Train SVM %%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
            
            foldC = C(c);
            model = cell(numClassifier,1);
            parfor i = 1 : numClassifier
                model{i} = svmtrain(TrainingGroup(:,i),[trainLabel',trainKernel],sprintf('-t 4 -b 1 -c %f -q',foldC));
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%% 3. Use SVM %%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            fprintf('Predicting...');
            parfor i = 1 : numClassifier
                [label, acc, p] =svmpredict(SampleGroup(:,i), [sampleLabel',sampleKernel], model{i}, '-b 1');
                accList(i,c) = accList(i,c) + sum(label == SampleGroup(:,i))/ numel(SampleGroup(:,i));
            end
        end
    end
    %% generate outputs
    % CBest
    if fUniqueC
        accSumList = mean(accList,1);
        [~,temp] = max(accSumList);
        CBest = C(temp);
    else
        [~,temp] = max(accList,[],2);
        CBest = zeros(numClassifier,1);
        for i = 1 : numClassifier
            CBest(i) = C(temp(i));
        end
    end
    
    % Model & gammaOut
    if fNeedModel
        if ~fPrecomputed
            % normal training
            parfor i = 1 : numClassifier
                if fUniqueC
                    svmModels{i} = svmtrain(trainGroup,trainFtr,sprintf('-t 4 -b 1 -c %f -q',CBest));
                else
                    svmModels{i} = svmtrain(trainGroup,trainFtr,sprintf('-t 4 -b 1 -c %f -q',CBest(i)));
                end
            end
        else
            % precomputed kernel training
            distMx = pdist2(trainFtr,trainFtr).^2;
            gammaOut = 1/mean(mean(distMx));
            trainKernel = exp(-gammaOut*distMx);
            trainLabel = (1:1:size(trainKernel,1));
            parfor i = 1 : numClassifier
                if fUniqueC
                    svmModels{i} = svmtrain(trainGroup(:,i),[trainLabel',trainKernel],sprintf('-t 4 -b 1 -c %f -q',CBest));
                else
                    svmModels{i} = svmtrain(trainGroup(:,i),[trainLabel',trainKernel],sprintf('-t 4 -b 1 -c %f -q',CBest(i)));
                end
            end
        end
    end
end
