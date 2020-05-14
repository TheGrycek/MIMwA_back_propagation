E=10^-9;
wektor_wej=[-1 1; -1 -1; 1 -1; 1 1]';
wektor_wag_hid=rand(1,2)';
wektor_wag_hid_zapis=wektor_wag_hid';
wektor_wag_out=rand(1,1)';
wektor_wag_out_zapis=wektor_wag_out';
wektor_bias_hid=rand(1,1)';
wektor_bias_hid_zapis=wektor_bias_hid;
wektor_bias_out=rand(1,1)';
wektor_bias_out_zapis=wektor_bias_out;
wektor_uczacy=[0 0.9 0.9 0];

index=randperm(length(wektor_wej));
        
        for i=1:length(wektor_wej)
            wektor_wej(i)=wej_backup(index(i));
            wektor_uczacy(i)=targ_backup(index(i));
        end

[wektor_wag_hid, wektor_wag_out, wektor_bias_hid, wektor_bias_out]=Backprop_alg(E, wektor_wej, wektor_uczacy, wektor_wag_hid, wektor_wag_out, wektor_bias_hid, wektor_bias_out);


wektor_wag_hid=wektor_wag_hid';
wektor_wag_out=wektor_wag_out';
wektor_bias_hid=wektor_bias_hid';
wektor_bias_out=wektor_bias_out';

disp('Wagi warstwy ukrytej przed uczeniem:')
x1=['bias=',num2str(wektor_bias_hid_zapis),' , w1=',num2str(wektor_wag_hid_zapis(:,1)),' , w2=',num2str(wektor_wag_hid_zapis(:,2))];
disp(x1)
disp('Wagi warstwy ukrytej po uczeniu:')
x2=['bias=',num2str(wektor_bias_hid),' , w1=',num2str(wektor_wag_hid(:,1)),' , w2=',num2str(wektor_wag_hid(:,2))];
disp(x2)
disp('Wagi warstwy wyjœciowej przed uczeniem:')
x3=['bias=',num2str(wektor_bias_out_zapis),' , w1=',num2str(wektor_wag_out_zapis)];
disp(x3)
disp('Wagi warstwy wyjœciowej po uczeniu:')
x4=['bias=',num2str(wektor_bias_out),' , w1=',num2str(wektor_wag_out)];
disp(x4)
disp('Wektor danych wejœciowych:')
disp(wektor_wej)
disp('Wektor danych ucz¹cych:')
disp(wektor_uczacy)
disp('Wektor odpowiedzi perceptronu:')
disp(y)

