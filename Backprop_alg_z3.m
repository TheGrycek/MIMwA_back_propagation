function [wektor_wag_hid, wektor_wag_out, wektor_bias_hid, wektor_bias_out] = Backprop_alg_z3(E, wektor_wej, wektor_uczacy, wektor_wag_hid, wektor_wag_out, wektor_bias_hid, wektor_bias_out)
    
    k=0;
    Esum=1;
    n=3*10^-3;
    n_wyj=10^-3;
    n_b=10^-3;
    
    nx=[];
    ny=[];
    
    figure;
        
   while Esum>E && k<1000
 
       index=randperm(length(wektor_wej));
       
       
       k=k+1;
        
        wej_backup=wektor_wej;
        targ_backup=wektor_uczacy;
        
        for i=1:length(wektor_wej)
            wektor_wej(i)=wej_backup(index(i));
            wektor_uczacy(i)=targ_backup(index(i));
        end
        
        Esum=0;
        y1=[];
        y2=[];
        
        for z=1:length(wektor_wag_out)
            y1=[y1; perceptron_tansig(wektor_wej,wektor_wag_hid(:,z),wektor_bias_hid(z))]; %1 warstwa
        end
        y2=perceptron_lin(y1,wektor_wag_out,wektor_bias_out); %2 warstwa

        e_caly=wektor_uczacy-y2;
        
        for i=1:length(e_caly)
            Esum=Esum+0.5*e_caly(i)^2;%b³¹d sieci
        end
        
        %warstwa hidden
       
       
            for i=1:length(wektor_wej)
                
                y1=[];
        y2=[];
                
                 for z=1:length(wektor_wag_out)
                    y1=[y1; perceptron_tansig(wektor_wej,wektor_wag_hid(:,z),wektor_bias_hid(z))]; %1 warstwa
                 end
                y2=perceptron_lin(y1,wektor_wag_out,wektor_bias_out); %2 warstwa
                
                
                for z=1:length(wektor_wag_out)
                    nx=[nx  e_caly(i)*(1-(tansig(wektor_wag_hid(:,z)'*wektor_wej(:,i)+wektor_bias_hid(z))).^2)*wektor_wej(:,i)];
                end
                dw_h= nx;
                nx=[];
                wektor_wag_hid=wektor_wag_hid + n*dw_h;
                
                for j=1:1
                    for z=1:length(wektor_wag_out)
                       ny=[ny  n_b*(wektor_wag_hid(j,z)*e_caly(i)*(1-tansig(wektor_wag_hid(j,z)*wektor_wej(j,i)+wektor_bias_hid(z)).^2))]; %œrednia b³êdu
                    end
                        wektor_bias_hid=wektor_bias_hid+ny';
                        ny=[];
                    
                end
                
                
                
                
        
            dw_o=e_caly(i)*y1(:,i);
            wektor_bias_out=wektor_bias_out+n_b*e_caly(i);
        
        wektor_wag_out=wektor_wag_out + n_wyj*dw_o;
                
            end
       
        
        
        %warstwa output
        
        
        wektor_wej=wej_backup;
        wektor_uczacy=targ_backup;
        
        pause(0.00001)
        
        y1=[];
        y2=[];
        
        for z=1:length(wektor_wag_out)
            y1=[y1; perceptron_tansig(wektor_wej,wektor_wag_hid(:,1),wektor_bias_hid(z))]; %1 warstwa
        end
        y2=perceptron_lin(y1,wektor_wag_out,wektor_bias_out); %2 warstwa

        
        clf
        
        hold on
        p1=plot(wektor_uczacy,'k');
        p2=plot(y2,'r');

        
    end
    y1
    y2
    wektor_uczacy
    k

end