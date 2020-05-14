E=10^-5;

[x,t]=simplefit_dataset;
x=mapminmax(x);
t=mapminmax(t);


wejscia=x;
wektor_uczacy=t';
wektor_wag_hid=rand(1,length(t))';
wektor_wag_out=rand(1,length(t))';
wektor_bias_hid=rand(1,length(t))';
wektor_bias_out=rand(1,length(t))';


[wektor_wag_hid, wektor_wag_out, wektor_bias_hid, wektor_bias_out]=Backprop_alg_z4(E, wejscia, t, wektor_wag_hid, wektor_wag_out, wektor_bias_hid, wektor_bias_out);
% wektor_wag_hid
% wektor_wag_out
% wektor_bias_hid
% wektor_bias_out