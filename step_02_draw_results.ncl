;*************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"
;*************************************************

begin

fontheight = 0.018
plottype = "oldps"

exp_name = "EXP_NAME"

seq_in = 6
seq_out = 6
intv = 60    ; time interval between sequences, [minutes]


; Get predicted data
ckpt_name="CKPT_NAME"
;dpath="/data/Deeplearning/predict_rainrate/01_train_model/output/predict/"+exp_name+"/"
dpath="PRED_DIR"
dfile="predict_"+ckpt_name+"_"+exp_name+".nc"

; Set image output directory
;outpath="/data/Deeplearning/predict_rainrate/01_train_model/output/image/"+exp_name+"/"
outpath="IMAGE_DIR"+"/"

; Set Radar data directory
rdr_path_root="./data/"


; Read predicted results
in = addfile(dpath+"/"+dfile, "r")
case = in->case       ; Initial date for each case 
tstep = in->time_step ; sequential indices for predicted sequences
date_list = in->date_list ; date list in [case, time_step]
pred_all = in->pred_rainrate    ; Predicted RDR rainrate

; Read LAT, LON data from RDR_downscale_1h_avg data
in_ll = addfile(rdr_path_root+"RDR_downscale_1h_avg_CAPPI/2012/20120601/RDR_avg1h_128_4km_2012-06-01_01:00:00.nc", "r")
lat2d = in_ll->XLAT
lon2d = in_ll->XLON


ndim = dimsizes(pred_all)
print(ndim)

; Get sample number from [case]
do k = 0, ndim(0)-1     ; Do-loop for cases(or samples)
; seq_date_temp = new( seq_in+seq_out+1, "string")
 seq_date = new( seq_in+seq_out, "string")   ; In my exp, data for initial time has not been used.
 seq_date!0 = "seq"
 seq_date&seq = ispan(0,seq_in+seq_out-1,1)

; seq_date_temp(0) = case(k)
 seq_date(0) = tostring(case(k))
 do j = 1, seq_in+seq_out-1
  seq_yy = str_get_cols(tostring(seq_date(j-1)),0,3)
  seq_mm = str_get_cols(tostring(seq_date(j-1)),4,5)
  seq_dd = str_get_cols(tostring(seq_date(j-1)),6,7)
  seq_HH = str_get_cols(tostring(seq_date(j-1)),8,9)
  seq_MM = str_get_cols(tostring(seq_date(j-1)),10,11)
  seq_date(j)=systemfunc("date -d '"+seq_yy+seq_mm+seq_dd+" "+seq_HH+":"+seq_MM+" "+intv+" minutes' +%Y%m%d%H%M")
 end do

;=================================================
;- Reducing the number of resulted figures.
 seq_init_HH = toint(str_get_cols(seq_date(0),8,9))
;- open wks at 0, 3, 6, 9, 12, 15, 18, 21 hours
 if ( seq_init_HH%3 .ne. 0 ) then
;- open wks at 1, 7, 13, 19 hours
; if ( seq_init_HH%6 .ne. 1 ) then
  continue
 end if
;-------------------------------------------------- 

;wksname = "results_"+exp_name+"_"+case(k)+"_KST"
wksname = ckpt_name+"_"+exp_name+"_"+case(k)+"_KST"
wks = gsn_open_wks(plottype, wksname)

 map = new( seq_in+3*seq_out, graphic)
 plot_in = new( seq_in, graphic)      ; plot for input sequence
 plot_rdr = new( seq_out, graphic)    ; plot for ground truth
 plot_pred = new( seq_out, graphic)   ; plot for output sequence
 plot_diff = new( seq_out, graphic)   ; plot for difference fields


 mres = True
 mres@gsnDraw = False
 mres@gsnFrame = False
 mres@mpOutlineOn = True
 mres@mpOutlineBoundarySets = "Geophysical"
 mres@mpProjection = "LambertConformal"
 mres@mpLambertParallel1F = 30
 mres@mpLambertParallel2F = 60
 mres@mpLambertMeridianF  = 126
 mres@mpLimitMode = "Corners"
 mres@mpLeftCornerLatF = lat2d(0,0)
 mres@mpLeftCornerLonF = lon2d(0,0)
 mres@mpRightCornerLatF = lat2d(ndim(2)-1, ndim(3)-1)
 mres@mpRightCornerLonF = lon2d(ndim(2)-1, ndim(3)-1)
 mres@mpGridAndLimbOn = True
 mres@mpGridLineDashPattern = 0
 mres@mpGridLineThicknessF = 0.2
 mres@mpGridLatSpacingF = 1.0
 mres@mpGridLonSpacingF = 1.0
 mres@mpGeophysicalLineThicknessF = 1.0
 mres@mpDataBaseVersion = "Ncarg4_1"
 mres@mpDataSetName     = "Earth..4"
 mres@mpFillAreaSpecifiers = (/"Water", "North Korea", "South Korea", "Japan", "China" /)
 mres@mpSpecifiedFillColors = (/ -1, -1, -1, -1, -1 /)
 mres@pmTickMarkDisplayMode = "Always"
 mres@tmXBMajorLengthF        = 0.000
 mres@tmXTMajorLengthF        = 0.000
 mres@tmYLMajorLengthF        = 0.000
 mres@tmYRMajorLengthF        = 0.000
 mres@tmXTOn                  = False
 mres@tmXBOn                  = True
 mres@tmYLOn                  = True
 mres@tmYROn                  = False
 mres@tmXBLabelFontHeightF    = fontheight * 0.8
 mres@tmYLLabelFontHeightF    = fontheight * 0.8
 mres@tmXBLabelDeltaF = -0.5
 mres@tmYLLabelDeltaF = -0.5
 mres@tmXBLabelStride         = 2
 mres@tmYLLabelStride         = 2

 do i = 0, seq_in+3*seq_out-1
  map(i) = gsn_csm_map(wks, mres)
 end do


 do j = 0, seq_in+seq_out-1
  cyy = str_get_cols(seq_date(j),0,3)
  cmm = str_get_cols(seq_date(j),4,5)
  cdd = str_get_cols(seq_date(j),6,7)
  cHH = str_get_cols(seq_date(j),8,9)
  cMM = str_get_cols(seq_date(j),10,11)

  if ( toint(cyy).ge.2012 .and. toint(cyy).lt.2016) then
   rdr_path = rdr_path_root+"RDR_downscale_1h_avg_CAPPI/"+cyy+"/"+cyy+cmm+cdd+"/"
  else if ( toint(cyy).ge.2016 .and. toint(cyy).lt.2020) then
   rdr_path = rdr_path_root+"RDR_downscale_1h_avg_HSR/"+cyy+"/"+cyy+cmm+cdd+"/"
  end if
  end if

  rdr_file = "RDR_avg1h_128_4km_"+cyy+"-"+cmm+"-"+cdd+"_"+cHH+":"+cMM+":00.nc"

  in_rdr = addfile(rdr_path+rdr_file, "r")
  cappi = in_rdr->rain1h     ; input sequence or ground truth
  cappi@lat2d = lat2d
  cappi@lon2d = lon2d

  if ( j.ge.seq_in ) then
   predict = pred_all(k, j-seq_in, :, :)  ; get prediction results for corresponding [seq_date]
   predict@lat2d = lat2d
   predict@lon2d = lon2d
   diff = predict - cappi                 ; difference field for corresponding [seq_date]
   diff@lat2d = lat2d
   diff@lon2d = lon2d
  end if


  ; set common option 
  res = True
  res@gsnDraw = False
  res@gsnFrame = False
  res@gsnLeftString = ""
  res@gsnLeftStringFontHeightF = fontheight * 2.00
  res@gsnRightString = cyy+"-"+cmm+"-"+cdd+" "+cHH+":"+cMM+" KST"
  res@gsnRightStringFontHeightF = fontheight * 2.00

  ; set options for rainrate field
  opt0   = res
  opt0@cnLinesOn = False
  opt0@cnLineLabelsOn = False
  opt0@cnFillOn = True
  opt0@cnFillPalette= "wskim_rainrate"
  opt0@cnLevelSelectionMode = "ExplicitLevels"
  opt0@cnLevels = (/  0.1, 0.2, 0.4, 0.6, 0.8,   1,  \
                      1.5,   2,   3,   4,   5,   6,   7,   8,   9,  10, \
                       12,  14,  16,  18,  20,  25,  30,  35,  40,  \
                       50,  60,  70,  80,  90, 100  /)
  opt0@cnFillColors = (/  -1,   1,   2,   3,   4,   5,   6, \
                           7,   8,   9,  10,  11,  12,  13,  14,  15,  16, \
                          17,  18,  19,  20,  21,  22,  23,  24,  25, \
                          26,  27,  28,  29,  30,  31 /) 
  ;opt0@cnMissingValFillColor = NhlNewColor(wks, 0.2, 0.2, 0.2)
  opt0@cnMissingValFillColor = -1
  opt0@cnInfoLabelOn = False
  opt0@lbLabelBarOn = False
  opt0@lbLabelStride = 1
  opt0@lbLabelFontHeightF = fontheight * 0.60
  opt0@lbLabelPosition = "Right"
  opt0@lbTitleOn = True
  opt0@lbTitleString = "mm h~S~-1~N~"
  opt0@lbTitleFontHeightF = fontheight * 0.7
  opt0@lbTitlePosition = "Top"
  opt0@lbTitleDirection = "Across"
  opt0@lbTitleOffsetF   = -0.05
  opt0@lbOrientation = "Vertical"
  opt0@pmLabelBarWidthF = 0.08
  opt0@pmLabelBarHeightF = 0.65
  opt0@pmLabelBarOrthogonalPosF = 0.02
  opt0@pmLabelBarParallelPosF = 0.55


  ; set options for difference field
  opt1   = res
  opt1@cnLinesOn = False
  opt1@cnLineLabelsOn = False
  opt1@cnFillOn  = True
  opt1@cnLevelSelectionMode = "ExplicitLevels"
  opt1@cnLevels     =  (/   -120, -80, -40, -20, -10,  -5,  -2,   0,  \
                               2,   5,  10,  20,  40,  80,  120   /)
  opt1@cnFillColors = (/    1,   2,   3,   4,   5,   6,   7,   8, \
                            9,  10,  11,  12,  13,  14,  15,  16  /) 
  opt1@cnFillPalette = "BlueDarkRed18"
  opt1@cnInfoLabelOn = False
  opt1@lbLabelBarOn = False
  opt1@lbLabelStride = 1
  opt1@lbLabelFontHeightF = fontheight * 0.65
  opt1@lbLabelPosition = "Right"
  opt1@lbTitleOn = True
  opt1@lbTitleString = "hPa"
  opt1@lbTitleFontHeightF = fontheight * 0.7
  opt1@lbTitlePosition = "Top"
  opt1@lbTitleDirection = "Across"
  opt1@lbTitleOffsetF   = -0.05
  opt1@lbOrientation = "Vertical"
  opt1@pmLabelBarWidthF = 0.08
  opt1@pmLabelBarHeightF = 0.65
  opt1@pmLabelBarOrthogonalPosF = 0.02
  opt1@pmLabelBarParallelPosF = 0.55


  if ( j.ge.0 .and. j.lt.seq_in) then
   plot_in(j) = gsn_csm_contour(wks, cappi(:,:), opt0)
   overlay(map(j),plot_in(j))
  else if ( j.ge.seq_in .and. j.lt.seq_in+seq_out) then
   plot_rdr(j-seq_in) = gsn_csm_contour(wks, cappi(:,:), opt0)
   plot_pred(j-seq_in) = gsn_csm_contour(wks, predict(:,:), opt0)
   plot_diff(j-seq_in) = gsn_csm_contour(wks, diff(:,:), opt1)
   overlay(map(j),plot_rdr(j-seq_in))
   overlay(map(j+seq_out),plot_pred(j-seq_in))
   overlay(map(j+2*seq_out),plot_diff(j-seq_in))
   delete(predict)
   delete(diff)
  end if
  end if
 delete(cappi)
 end do 

;======================
; Create panel
;----------------------
 tyy = str_get_cols(tostring(case(k)),0,3)
 tmm = str_get_cols(tostring(case(k)),4,5)
 tdd = str_get_cols(tostring(case(k)),6,7)
 tHH = str_get_cols(tostring(case(k)),8,9)
 tMM = str_get_cols(tostring(case(k)),10,11)
 tires     = True
 tires@txFontHeightF = 0.018
 tires@txFont = 22
 gsn_text_ndc(wks, "CASE: "+tyy+"-"+tmm+"-"+tdd+"_"+tHH+":"+tMM+" KST", 0.5, 0.95, tires)


 resp0     = True
 resp0@gsnFrame = False
 resp0@gsnPanelBottom = 0.250
 resp0@gsnPanelLabelBar = True
 resp0@lbTitleOn = True
 resp0@lbTitleString = "mm h~S~-1~N~"
 resp0@lbTitleFontHeightF = fontheight * 0.6
 resp0@lbTitlePosition = "Right"
 resp0@lbTitleDirection = "Across"
 resp0@lbTitleOffsetF = -0.050
 resp0@lbLabelStride = 1
 resp0@lbLabelFontHeightF = fontheight * 0.6
 resp0@pmLabelBarWidthF = 0.95
 resp0@pmLabelBarHeightF = 0.045
 resp0@pmLabelBarParallelPosF = 0.05

 resp1     = True
 resp1@gsnFrame = False
 resp1@gsnPanelTop = 0.450
 resp1@gsnPanelLabelBar = True
 resp1@lbTitleOn = True
 resp1@lbTitleString = "mm h~S~-1~N~"
 resp1@lbTitleFontHeightF = fontheight * 0.6
 resp1@lbTitlePosition = "Right"
 resp1@lbTitleDirection = "Across"
 resp1@lbTitleOffsetF = -0.050
 resp1@lbLabelStride = 1
 resp1@lbLabelFontHeightF = fontheight * 0.6
 resp1@pmLabelBarWidthF = 0.95
 resp1@pmLabelBarHeightF = 0.045
 resp1@pmLabelBarParallelPosF = 0.05

 gsn_panel(wks, map(0:seq_in+2*seq_out-1), (/3,max((/seq_in,seq_out/))/), resp0)
 gsn_panel(wks, map(seq_in+2*seq_out:seq_in+3*seq_out-1), (/1,max((/seq_in,seq_out/))/), resp1)

 system("ps2eps -f "+wksname+".ps")
 system("eps2png -width 1800 "+wksname+".eps")
 system("rm -rf "+wksname+"*ps")
 system("mv "+wksname+".png "+outpath)

 print("=== Done plotting: "+sprinti("%4.0i",k)+"/"+sprinti("%4.0i",ndim(0)-1)+" ===")
end do




end
