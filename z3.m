E=8^-2;
Esum=1;
k=0;
n=10^-2;
figure
z=0;

[x,t]=simplefit_dataset;
x=mapminmax(x);
t=mapminmax(t);


wektor_wej=x;
wektor_uczacy=t;

wektor_wag_hid=rand(30,1);
wektor_wag_hid_zapis=wektor_wag_hid';
wektor_wag_out=rand(1,30);
wektor_wag_out_zapis=wektor_wag_out';
wektor_bias_hid=rand(1,30);
wektor_bias_hid_zapis=wektor_bias_hid';
wektor_bias_out=rand(1,1);
wektor_bias_out_zapis=wektor_bias_out';

wej_backup=wektor_wej;
targ_backup=wektor_uczacy;

while Esum>E && k<=100000
    
     k=k+1;
     Esum=0;
     e_caly=[];
     y1=[];
     y2=[];
     
     index=randperm(length(wektor_wej));
            
     wektor_wej=wektor_wej(index);
     wektor_uczacy=wektor_uczacy(index);
     
     
     
     %pêtla g³ówna propagacji
     for i=1:length(wektor_wej)
         
         %propagacja w przód
         y1=[y1 tansig(wektor_wag_hid*wektor_wej(:,i)+wektor_bias_hid')];
         y2=[y2 wektor_wag_out*y1(:,i)+wektor_bias_out']; 

         %propagacja wsteczna + aktualizacja wag
        e_caly(i)=wektor_uczacy(i)-y2(i);
        
        
     %warstwa ukryta
         %zmiana wag
         e_hid=e_caly(i) * (1-tansig(wektor_wag_hid'*wektor_wej(:,i)+wektor_bias_hid).^2);
         dw_hid=n * e_hid * wektor_wej(:,i);
         wektor_wag_hid=wektor_wag_hid+dw_hid';
         
         %zmiana biasów
         db_hid=n * e_hid;
         wektor_bias_hid=wektor_bias_hid + db_hid;
         
         
     %warstwa koñcowa
         %zmiana wag
         dw_out=n * e_caly(i) * y1(:,i);
         wektor_wag_out=wektor_wag_out + dw_out';
         
         %zmiana biasów
         db_out=n * e_caly(i);
         wektor_bias_out=wektor_bias_out+db_out;
     end
     
     
     z=z+1;
     if z==100
         y1=[];
         y2=[];
         for i=1:length(wej_backup)
            y1=[y1 tansig(wektor_wag_hid*wej_backup(:,i)+wektor_bias_hid')];
            y2=[y2 wektor_wag_out*y1(:,i)+wektor_bias_out'];   
         end

         
            pause(0.001)
            clf
            hold on
            
            xlim([0 94]);
            ylim([-1 1]);
            p1=plot(targ_backup,'k');
            p2=plot(y2,'r');
            legend('show');
            legend({'Wektor Uczacy','Wyjœcie Sieci'})
%             txt=['Iteracja:' int2str(k)];
%             text(1,0.9,txt)
            z=0;

     end
     
     %wyliczenie b³êdu
     for i=1:length(e_caly)
         Esum=Esum+0.5*e_caly(i)^2;%b³¹d sieci
     end    
end


disp('Wektor wag warstwy ukrytej przed uczeniem:')
x1=[num2str(wektor_wag_hid_zapis)];
disp(x1)
disp('Wektor wag warstwy ukrytej po uczeniu:')
x2=[num2str(wektor_wag_hid')];
disp(x2)
disp('Wektor biasów warstwy ukrytej przed uczeniem:')
x3=[num2str(wektor_bias_hid_zapis')];
disp(x3)
disp('Wektor biasów warstwy ukrytej po uczeniu:')
x4=[num2str(wektor_bias_hid)];
disp(x4)

disp('Wektor wag warstwy wyjœciowej przed uczeniem:')
x5=[num2str(wektor_wag_out_zapis')];
disp(x5)
disp('Wektor wag warstwy wyjœciowej po uczeniu:')
x6=[num2str(wektor_wag_out)];
disp(x6)
disp('Wektor biasów warstwy wyjœciowej przed uczeniem:')
x7=[num2str(wektor_bias_out_zapis)];
disp(x7)
disp('Wektor biasów warstwy wyjœciowej po uczeniu:')
x4=[num2str(wektor_bias_out)];
disp(x4)

disp('Wektor danych wejœciowych:')
disp(wektor_wej)
disp('Wektor danych ucz¹cych:')
disp(wektor_uczacy)
disp('Wektor odpowiedzi sieci neuronowej:')
disp(y2)

