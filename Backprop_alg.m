function [wektor_wag_hid, wektor_wag_out, wektor_bias_hid, wektor_bias_out] = Backprop_alg(E, wektor_wej, wektor_uczacy, wektor_wag_hid, wektor_wag_out, wektor_bias_hid, wektor_bias_out)
    
    z=0;
    Esum=1;
    n=2*10^-1;
    figure

    while Esum>E && z<=100000
        z=z+1;
        Esum=0;

        y1=perceptron_tansig(wektor_wej,wektor_wag_hid,wektor_bias_hid); %1 warstwa
        y2=perceptron_tansig(y1,wektor_wag_out,wektor_bias_out); %2 warstwa

        e_caly=wektor_uczacy-y2;
        for i=1:length(e_caly)
            Esum=Esum+0.5*e_caly(i)^2;%b³¹d sieci
        end
        
        %warstwa ukryta
        dw_h=0;
        for i=1:length(wektor_wej)
            dw_h=dw_h + e_caly(i)*(1-tansig(wektor_wag_hid'*wektor_wej(:,i)+wektor_bias_hid).^2)*wektor_wej(:,i);
            for j=1:length(wektor_wag_hid)
                wektor_bias_hid=wektor_bias_hid+n*0.5*(wektor_wag_hid(j)*e_caly(i)*(1-tansig(wektor_wag_hid(j)*wektor_wej(j,i)+wektor_bias_hid).^2)); %œrednia b³êdu
            end
        end
        wektor_wag_hid=wektor_wag_hid + n*dw_h;
        
        %warstwa output
        dw_o=0;
        for i=1:length(wektor_wej)
            dw_o=dw_o + e_caly(i)*y1(i);
            wektor_bias_out=wektor_bias_out+n*e_caly(i);
        end
        wektor_wag_out=wektor_wag_out + n*dw_o;
        
        pause(0.0001)
        clf
        hold on
        p1=plot(wektor_uczacy,'k','Linewidth',2);
        p2=plot(y2,'r');
        
    end

end