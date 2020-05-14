function [wektor_wag_hid, wektor_wag_out, wektor_bias_hid, wektor_bias_out] = Backprop_alg_z4(E, wektor_wej, wektor_uczacy, wektor_wag_hid, wektor_wag_out, wektor_bias_hid, wektor_bias_out)
    
    z=0;
    Esum=1;
    n=10^-1;
    perm_n=1*10^-1;
    
    figure;
    

    while Esum>E && z<=1000
        perm=(rand-0.5)*perm_n;
        
        wej_backup=wektor_wej;
        targ_backup=wektor_uczacy;
        
        wektor_wej=wektor_wej+perm;
        wektor_uczacy=wektor_uczacy+perm;
        
        z=z+1;
        Esum=0;

        y1=perceptron_tansig(wektor_wej,wektor_wag_hid,wektor_bias_hid); %1 warstwa
        y2=perceptron_lin(y1,wektor_wag_out,wektor_bias_out); %2 warstwa

        e_caly=wektor_uczacy-y2;
        for i=1:length(e_caly)
            Esum=Esum+0.5*e_caly(i)^2;%b³¹d sieci
        end
        
        %warstwa ukryta
        dw_h=[];
        for i=1:length(wektor_wag_hid)
            dw_h=[dw_h; wektor_wag_hid(i,1)*e_caly(i)*(1-tansig(wektor_wag_hid(i,1)*wektor_wej(1,i)+wektor_bias_hid(i)).^2)*wektor_wej(1,i)];
            wektor_bias_hid(i)=wektor_bias_hid(i)+n*wektor_wag_hid(i,1)*e_caly(i)*(1-tansig(wektor_wag_hid(i,1)*wektor_wej(1,i)+wektor_bias_hid(i)).^2); %œrednia b³êdu
        end
        wektor_wag_hid=wektor_wag_hid + n*dw_h;
        
        %warstwa output
        dw_o=[];
        for i=1:length(wektor_wag_out)
            dw_o=[dw_o; e_caly(i)*y1(i)];
            wektor_bias_out(i)=wektor_bias_out(i)+n*e_caly(i);
        end
        wektor_wag_out=wektor_wag_out + n*dw_o;
        
        wektor_wej=wej_backup;
        wektor_uczacy=targ_backup;
        
        pause(0.01)
        
        clf
        
        hold on
        xlim([0 100]);
        ylim([0 10]);
        p1=plot(wektor_uczacy,'k');
        p2=plot(y2,'r');
        
    end
    y1
    y2
    wektor_uczacy
    z

end