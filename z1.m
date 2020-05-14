df=@(x)(1-tansig(x).^2);

wej=[-1 1; -1 -1; 1 -1; 1 1]';
wag=rand(1,2)';
wag_zapis=wag';
bias=rand(1,1);
bias_zapis=bias;
ucz=[0 0 0.1 0.1];
n=10^-1;
Emax=10^-10;
E=1;
z=0;
figure
axis([1 4 -0.6 1])
while E>Emax && z<=1000
    z=z+1;
    y=perceptron_tansig(wej,wag,bias);
    
    E=0;
    em=[];
    for i=1:length(ucz)
        em=[em ucz(i) - y(i)];
        E=0.5*em(i)^2 + E;
    end
    dw=0;
    for j=1:length(wej)
        dw=dw + em(j)*df(wag'*wej(:,j)+bias)*wej(:,j);
        bias=bias+n*em(j);
    end
    wag=wag+n*dw;
    
    
        pause(0.1)
        clf
        hold on
        p1=plot(ucz,'k','Linewidth',2);
        p2=plot(y,'r');
end
wag=wag';

disp('Wagi przed uczeniem:')
x1=['bias=',num2str(bias_zapis),' , w1=',num2str(wag_zapis(:,1)),' , w2=',num2str(wag_zapis(:,2))];
disp(x1)
disp('Wagi po uczeniu:')
x2=['bias=',num2str(bias),' , w1=',num2str(wag(:,1)),' , w2=',num2str(wag(:,2))];
disp(x2)
disp('Wektor danych wejœciowych:')
disp(wej)
disp('Wektor danych ucz¹cych:')
disp(ucz)
disp('Wektor odpowiedzi perceptronu:')
disp(y)
disp('Liczba iteracji:')
disp(z)
