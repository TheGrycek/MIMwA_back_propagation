function y = perceptron_tansig(wej,wag,bias)
y=[];
    for i=1:length(wej(1,:))
    
        y=[y tansig(wag'*wej(:,i)+bias)];
    
    end
end