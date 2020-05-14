function y = perceptron_lin(wej,wag,bias)
y=[];
    for i=1:length(wej(1,:))
    
        y=[y wag'*wej(:,i)+bias];
    
    end
end