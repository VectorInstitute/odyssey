# GEMINI extract-dry report: `subdural_hematoma_v1_0_0`

## Per-object row counts and null fractions

### `admdad_subset` (rows: 2268000)

| column | null count |
| --- | --- |
| genc_id | <6 |
| country | 547000 |
| admitting_service_raw | 192000 |
| discharging_service_raw | 269000 |
| total_direct_cost | 1892000 |
| total_indirect_cost | 1892000 |
| total_cost | 1892000 |
| admit_category | <6 |
| discharge_disposition | <6 |
| responsibility_for_payment | <6 |
| province_territory_issuing_health_card_number | 1871000 |
| number_of_alc_days | 670000 |
| institution_from | 1223000 |
| institution_to | 884000 |
| readmission | 390000 |
| residence_code | 91000 |
| gender | <6 |
| age | <6 |
| mrp_service | 3000 |
| entry_code | <6 |
| admission_date_time | <6 |
| discharge_date_time | <6 |
| admitting_service_mapped | 458000 |
| discharging_service_mapped | 463000 |
| patient_id_hashed | 1000 |
| mrp_service_raw | 2094000 |
| patient_service_subservice | 2217000 |
| patient_service_subservice_mapped | 2227000 |
| mrp_service_mapped | 1489000 |
| admit_via_ambulance | 378000 |
| alc_service_transfer_flag | 420000 |
| blood_transfusion_indicator | 650000 |
| main_patient_service_raw | 1874000 |
| main_patient_service_mapped | 2133000 |
| row_num | <6 |
| hospital_num | <6 |

### `cohort` (rows: 2268000)

| column | null count |
| --- | --- |
| genc_id | <6 |

### `derived_variables_subset` (rows: 2268000)

| column | null count |
| --- | --- |
| genc_id | <6 |
| mlaps | 571000 |
| admit_charlson_derived | 0 |
| all_charlson_derived | 0 |
| los_days_derived | <6 |
| readmission_30d_derived | 372000 |
| readmission_7d_derived | 355000 |
| readmission_30d_derived_cihi | 502000 |
| readmission_7d_derived_cihi | 486000 |
| in_hospital_mortality_derived | <6 |
| icu_entry_derived | <6 |
| icu_entry_in_24hr_derived | 0 |
| icu_entry_in_48hr_derived | 0 |
| icu_entry_in_72hr_derived | 0 |
| icu_los_days_derived | 0 |
| icu_los_hrs_derived | 0 |
| n_img_xray_derived | <6 |
| n_img_ct_derived | <6 |
| n_img_mri_derived | <6 |
| n_img_us_derived | <6 |
| n_img_int_derived | <6 |
| n_rbc_transfusion_derived | 143000 |
| n_app_rbc_transfusion_derived | 143000 |
| n_routine_bloodwork_derived | <6 |
| from_acute_care_institution_derived | 734000 |
| to_acute_care_institution_derived | 728000 |
| covid_icd_confirmed_derived | 0 |
| covid_icd_suspected_derived | 0 |
| epicare | 30000 |
| mlaps_24hrs | 199000 |
| gim | <6 |
| all_med | <6 |
| hospital_num | <6 |

### `er_subset` (rows: 1859000)

| column | null count |
| --- | --- |
| genc_id | <6 |
| admit_via_ambulance | <6 |
| triage_level | 1000 |
| ambulance_arrival_date_time | 74000 |
| physician_initial_assessment_date_time | 3000 |
| triage_date_time | 1000 |
| disposition_date_time | 0 |
| left_er_date_time | 13000 |
| registration_date_time | 112000 |
| duration_er_stay_derived | 13000 |
| cacs | 366000 |
| ed_discharge_diagnosis | 956000 |
| institution_number | 348000 |
| institution_from | 1385000 |
| institution_to | 504000 |
| visit_disposition | 163000 |
| cacs_methodology_year | 173000 |
| cacs_riw | 232000 |
| cacs_riw_on | 373000 |
| mac_code | 161000 |
| blood_transfusion_indicator | 602000 |
| referral_source_prior_to_ambulatory_care_visit | 574000 |
| row_num | <6 |
| non_physician_initial_assessment_date_time | 471000 |
| hospital_num | <6 |

### `erconsults_subset` (rows: 1899000)

| column | null count |
| --- | --- |
| genc_id | <6 |
| consult_occurrence | 66000 |
| consult_service_code | 5000 |
| consult_service_description | 12000 |
| consult_request_date_time | 9000 |
| consult_arrival_date_time | 18000 |
| row_num | <6 |
| hospital_num | <6 |

### `erdiagnosis_subset` (rows: 5221000)

*Skipped per-column null-fraction check -- too large (>= 5,000,000 rows).*

### `erintervention_subset` (rows: 2936000)

| column | null count |
| --- | --- |
| genc_id | <6 |
| intervention_type | 666000 |
| intervention_code | 1000 |
| intervention_location_attribute | 572000 |
| intervention_status_attribute | 626000 |
| intervention_extent_attribute | 635000 |
| out_of_hospital_indicator | 2335000 |
| intervention_episode_start_date_time | 768000 |
| row_num | <6 |
| hospital_num | <6 |

### `ipcmg_subset` (rows: 2092000)

| column | null count |
| --- | --- |
| genc_id | <6 |
| methodology_year | 116000 |
| cmg | 0 |
| diagnosis_for_cmg_assignment | 416000 |
| cmg_intervention | 427000 |
| comorbidity_level | 9000 |
| riw_inpatient_atypical_indicator | 0 |
| riw | 46000 |
| row_num | <6 |
| hospital_num | <6 |

### `ipdiagnosis_subset` (rows: 13895000)

*Skipped per-column null-fraction check -- too large (>= 5,000,000 rows).*

### `iphig_subset` (rows: 2242000)

| column | null count |
| --- | --- |
| genc_id | <6 |
| hig_methodology_year | 171000 |
| hig_code | 40000 |
| hig_description | 2007000 |
| hig_atypical_code | 52000 |
| hig_atypical_code_desc | 2034000 |
| hig_weight | 9000 |
| hig_elos | 104000 |
| age_category | 125000 |
| homecare_flag | 9000 |
| scu_flag | 9000 |
| cardioversion_flag | 100000 |
| cell_saver_flag | 96000 |
| chemotherapy_flag | 94000 |
| dialysis_flag | 91000 |
| feeding_tube_flag | 95000 |
| heart_resuscitation_flag | 95000 |
| invasive_ventilation_ge_96h_flag | 103000 |
| invasive_ventilation_lt_96h_flag | 102000 |
| paracentesis_flag | 95000 |
| parenteral_nutrition_flag | 254000 |
| pleurocentesis_flag | 297000 |
| radiotherapy_flag | 95000 |
| tracheostomy_flag | 96000 |
| vascular_access_device_flag | 90000 |
| row_num | <6 |
| hospital_num | <6 |

### `ipintervention_subset` (rows: 4329000)

| column | null count |
| --- | --- |
| genc_id | <6 |
| intervention_type | 357000 |
| intervention_code | <6 |
| procedure_location | 3748000 |
| intervention_location_attribute | 734000 |
| intervention_status_attribute | 1252000 |
| intervention_extent_attribute | 880000 |
| intervention_episode_start_date_time | 8000 |
| row_num | <6 |
| hospital_num | <6 |

### `ipscu_subset` (rows: 632000)

| column | null count |
| --- | --- |
| genc_id | <6 |
| scu_admit_date_time | <6 |
| scu_discharge_date_time | <6 |
| icu_flag | <6 |
| row_num | <6 |
| scu_unit_number | <6 |
| hospital_num | <6 |

### `lab_subset` (rows: 659381000)

*Skipped per-column null-fraction check -- too large (>= 5,000,000 rows).*

### `locality_variables_subset` (rows: 2268000)

| column | null count |
| --- | --- |
| genc_id | <6 |
| da16uid | 43000 |
| da11uid | 43000 |
| version | 202000 |
| row_num | <6 |
| da21uid | 42000 |
| hospital_num | 19000 |

### `lookup_cci` (rows: 17000)

| column | null count |
| --- | --- |
| intervention_code | <6 |
| cci_short_title | <6 |
| cci_long_title | <6 |

### `lookup_ccsr` (rows: 1000)

| column | null count |
| --- | --- |
| ccsr | <6 |
| ccsr_desc | <6 |

### `lookup_cihi_codes` (rows: 1000)

| column | null count |
| --- | --- |
| table_name | <6 |
| column_name | <6 |
| value | <6 |
| description | <6 |
| version | <6 |

### `lookup_data_coverage` (rows: 1000)

| column | null count |
| --- | --- |
| data | <6 |
| min_date | 0 |
| max_date | 0 |
| hospital_num | <6 |
| additional_info | <6 |

### `lookup_hospital` (rows: 0)

| column | null count |
| --- | --- |
| institution_id | <6 |
| hospital_num | <6 |
| hospital_type | <6 |
| additional_info | <6 |
| gim_cohort_avail | <6 |
| other_med_cohort_avail | <6 |
| icu_cohort_avail | <6 |
| other_inpatient_cohort_avail | 0 |

### `lookup_icd10_ca_description` (rows: 19000)

| column | null count |
| --- | --- |
| diagnosis_code | <6 |
| short_description | <6 |
| long_description | <6 |
| type | <6 |
| version | <6 |

### `lookup_icd10_ca_to_ccsr` (rows: 82000)

| column | null count |
| --- | --- |
| diagnosis_code | <6 |
| ccsr_default | <6 |
| ccsr_1 | <6 |
| ccsr_2 | <6 |
| ccsr_3 | <6 |
| ccsr_4 | <6 |
| ccsr_5 | <6 |
| ccsr_6 | <6 |
| gemini_derived | <6 |
| ccsr_version | <6 |

### `lookup_lab_concept` (rows: 1000)

| column | null count |
| --- | --- |
| concept_id | <6 |
| vocabulary_id | <6 |
| concept_desc | <6 |

### `lookup_pharmacy_mapping` (rows: 15000)

| column | null count |
| --- | --- |
| search_type | <6 |
| raw_input | <6 |
| rxnorm_match | <6 |
| drug_group | <6 |
| project_name | <6 |
| last_updated | <6 |

### `lookup_pharmacy_route` (rows: 0)

| column | null count |
| --- | --- |
| route | <6 |
| route_administration | <6 |
| route_delivery | <6 |

### `lookup_statcan_v2016` (rows: 48000)

| column | null count |
| --- | --- |
| da16uid | <6 |
| c16_popdw_popdens_sqkm | 0 |
| c16_inc_limat | 2000 |
| c16_immcit | 0 |
| c16_immcit_cancit | 0 |
| c16_immcit_cancit_bel18 | 0 |
| c16_immcit_cancit_18up | 0 |
| c16_immcit_notcan | 0 |
| c16_immsta | 0 |
| c16_immsta_notimm | 0 |
| c16_immsta_imm | 0 |
| c16_immsta_imm5yrs | 0 |
| c16_immsta_nonpr | 0 |
| c16_ab | 0 |
| c16_ab_singab | 0 |
| c16_ab_singab_fn | 0 |
| c16_ab_singab_met | 0 |
| c16_ab_singab_in | 0 |
| c16_ab_multab | 0 |
| c16_ab_abnotelse | 0 |
| c16_ab_nonab | 0 |
| c16_vismin | 0 |
| c16_vismin_not | 0 |
| c16_eth_ab_nam | 0 |
| c16_eth_other_nam | 0 |
| c16_eth_eur | 0 |
| c16_eth_car | 0 |
| c16_eth_lat | 0 |
| c16_eth_afr | 0 |
| c16_eth_asi | 0 |
| c16_eth_oce | 0 |
| c16_ed_15over | 0 |
| c16_ed_15over_nocert | 0 |
| c16_ed_15over_secschool | 0 |
| c16_ed_15over_postsec | 0 |
| c16_ed_25to64 | 0 |
| c16_ed_25to64_nocert | 0 |
| c16_ed_25to64_secschool | 0 |
| c16_ed_25to64_postsec | 0 |
| c16_lab_ind_allind | 0 |
| c16_lab_ind | 0 |
| c16_lab_ind_notappl | 0 |
| c16_lab_occ | 0 |
| c16_lab_occ_notappl | 0 |
| c16_lab_occ_allocc | 0 |
| c16_lab_prh | 0 |
| c16_lab_prh_wfh | 0 |
| c16_lab_prh_out_of_can | 0 |
| c16_lab_prh_nfaddr_work | 0 |
| c16_lab_prh_usplace | 0 |
| c16_lab_class | 0 |
| c16_lab_class_allcl | 0 |
| c16_lab_class_allcl_empl | 0 |
| c16_lab_class_allcl_selfem | 0 |
| c16_lab_avg_weeks_worked | 0 |
| c16_lab_wact_work | 0 |
| c16_lab_wact_full | 0 |
| c16_lab_wact_part | 0 |
| c16_lab_wact_dnw | 0 |
| c16_lab_wact | 0 |
| c16_lab_labf | 0 |
| c16_lab_labf_inlf | 0 |
| c16_lab_labf_unempl | 0 |
| c16_lab_labf_empl | 0 |
| c16_lab_labf_notin | 0 |
| c16_lab_part_rate | 0 |
| c16_lab_empl_rate | 0 |
| c16_lab_unempl_rate | 0 |
| instability_da16 | 31000 |
| instability_q_da16 | 31000 |
| deprivation_da16 | 31000 |
| deprivation_q_da16 | 31000 |
| dependency_da16 | 31000 |
| dependency_q_da16 | 31000 |
| ethniccon_da16 | 31000 |
| ethniccon_q_da16 | 31000 |
| cm_atype | <6 |
| btippe | <6 |
| atippe | <6 |
| qabtippe | <6 |
| qnbtippe | <6 |
| dabtippe | <6 |
| dnbtippe | <6 |
| qaatippe | <6 |
| qnatippe | <6 |
| daatippe | <6 |
| dnatippe | <6 |
| impflg | <6 |
| c16_inc_total_u5_ct | 24000 |
| c16_inc_total_5to10_ct | 24000 |
| c16_inc_total_10to15_ct | 24000 |
| c16_inc_total_15to20_ct | 24000 |
| c16_inc_total_20to25_ct | 24000 |
| c16_inc_total_25to30_ct | 24000 |
| c16_inc_total_30to35_ct | 24000 |
| c16_inc_total_30to40_ct | 24000 |
| c16_inc_total_40to45_ct | 24000 |
| c16_inc_total_45to50_ct | 24000 |
| c16_inc_total_50to60_ct | 24000 |
| c16_inc_total_60to70_ct | 24000 |
| c16_inc_total_70to80_ct | 24000 |
| c16_inc_total_80to90_ct | 24000 |
| c16_inc_total_90to100_ct | 24000 |
| c16_inc_total_100to125_ct | 24000 |
| c16_inc_total_125to150_ct | 24000 |
| c16_inc_total_150to200_ct | 24000 |
| c16_inc_total_200up_ct | 24000 |
| c16_inc_num_ct | 24000 |
| ice_inc_c16_ct | 24000 |

### `lookup_statcan_v2021` (rows: 58000)

| column | null count |
| --- | --- |
| da21uid | <6 |
| c21_prov_code | <6 |
| c21_prov_name | <6 |
| c21_cd_code | <6 |
| c21_cd_name | <6 |
| c21_da_name | <6 |
| c21_popdw_pop21 | 0 |
| c21_popdw_pop16 | 58000 |
| c21_popdw_pop_perc_change | 58000 |
| c21_popdw_privdw_total | 0 |
| c21_popdw_privdw_usres | 0 |
| c21_popdw_popdens_sqkm | 0 |
| c21_land_area_sqkm | <6 |
| c21_inc_stats | 2000 |
| c21_inc_total_grps | 4000 |
| c21_inc_num | 4000 |
| c21_inc_med | 4000 |
| c21_inc_total | 4000 |
| c21_inc_total_u5 | 4000 |
| c21_inc_total_5to10 | 4000 |
| c21_inc_total_10to15 | 4000 |
| c21_inc_total_100up | 4000 |
| c21_inc_total_90to100 | 4000 |
| c21_inc_total_70to80 | 4000 |
| c21_inc_total_60to70 | 4000 |
| c21_inc_total_50to60 | 4000 |
| c21_inc_total_45to50 | 4000 |
| c21_inc_total_15to20 | 4000 |
| c21_inc_total_20to25 | 4000 |
| c21_inc_total_100to125 | 4000 |
| c21_inc_total_125to150 | 4000 |
| c21_inc_total_150to200 | 4000 |
| c21_inc_total_200up | 4000 |
| c21_inc_med_total_fam | 4000 |
| c21_inc_aft_tax_med | 4000 |
| c21_prev_inc_limat | 4000 |
| c21_inc_licoat | 4000 |
| c21_prev_inc_licoat | 9000 |
| c21_incavg_num | 4000 |
| c21_incavg_avg | 4000 |
| c21_immcit | 2000 |
| c21_immcit_cancit | 2000 |
| c21_immcit_cancit_bel18 | 2000 |
| c21_immcit_cancit_18up | 2000 |
| c21_immcit_notcan | 2000 |
| c21_immsta | 2000 |
| c21_immsta_notimm | 2000 |
| c21_immsta_imm | 2000 |
| c21_immsta_imm_11to21 | 2000 |
| c21_immsta_imm_11to15 | 2000 |
| c21_immsta_imm_16to21 | 2000 |
| c21_immsta_nonpr | 2000 |
| c21_ind | 2000 |
| c21_ind_id | 2000 |
| c21_ind_id_singind | 2000 |
| c21_ind_id_singind_fn | 2000 |
| c21_ind_id_singind_met | 2000 |
| c21_ind_id_singind_inuit | 2000 |
| c21_ind_id_multind | 2000 |
| c21_ind_id_indnotelse | 2000 |
| c21_ind_id_nonind | 2000 |
| c21_vismin | 2000 |
| c21_vismin_not | 2000 |
| c21_eth_ab_nam | 2000 |
| c21_eth_eur | 2000 |
| c21_eth_asi | 2000 |
| c21_eth_afr | 2000 |
| c21_eth_lat | 2000 |
| c21_eth_car | 2000 |
| c21_eth_other_nam | 2000 |
| c21_eth_cult_orig | 2000 |
| c21_ed_15over | 2000 |
| c21_ed_15over_nocert | 2000 |
| c21_ed_15over_secschool | 2000 |
| c21_ed_15over_postsec | 2000 |
| c21_ed_25to64 | 2000 |
| c21_ed_25to64_nocert | 2000 |
| c21_ed_25to64_secschool | 2000 |
| c21_ed_25to64_postsec | 2000 |
| c21_lab_occ | 2000 |
| c21_lab_occ_notappl | 2000 |
| c21_lab_occ_allocc | 2000 |
| c21_lab_ind | 2000 |
| c21_lab_ind_allind | 2000 |
| c21_lab_ind_notappl | 2000 |
| c21_lab_class | 2000 |
| c21_lab_class_allcl | 2000 |
| c21_lab_class_allcl_empl | 2000 |
| c21_lab_class_allcl_selfem | 2000 |
| c21_lab_wact | 2000 |
| c21_lab_wact_work | 2000 |
| c21_lab_wact_dnw | 2000 |
| c21_lab_wact_full | 2000 |
| c21_lab_wact_part | 2000 |
| c21_lab_avg_weeks_worked | 2000 |
| c21_lab_labf | 2000 |
| c21_lab_labf_inlf | 2000 |
| c21_lab_labf_notin | 2000 |
| c21_lab_labf_empl | 2000 |
| c21_lab_labf_unempl | 2000 |
| c21_lab_part_rate | 2000 |
| c21_lab_empl_rate | 2000 |
| c21_lab_unempl_rate | 2000 |
| c21_lab_plw | 2000 |
| c21_lab_plw_wfh | 2000 |
| c21_lab_plw_out_of_can | 2000 |
| c21_lab_plw_nfaddr_work | 2000 |
| c21_lab_plw_usplace | 2000 |
| c21_inc_total_25to30 | 4000 |
| c21_inc_total_30to35 | 4000 |
| c21_inc_total_40to45 | 4000 |
| c21_inc_total_35to40 | 4000 |
| Pop2021 | 38000 |
| households_dwellings_DA21 | 38000 |
| material_resources_DA21 | 38000 |
| age_labourforce_DA21 | 38000 |
| racialized_NC_pop_DA21 | 38000 |
| households_dwellings_q_DA21 | 38000 |
| material_resources_q_DA21 | 38000 |
| age_labourforce_q_DA21 | 38000 |
| racialized_NC_pop_q_DA21 | 38000 |
| btippe | 9000 |
| atippe | 9000 |
| qabtippe | 9000 |
| qnbtippe | 9000 |
| dabtippe | 9000 |
| dnbtippe | 9000 |
| qaatippe | 9000 |
| qnatippe | 9000 |
| daatippe | 9000 |
| dnatippe | 9000 |
| impflg | 9000 |
| popctrraclass | 9000 |

### `lookup_transfer_subset` (rows: 1250000)

| column | null count |
| --- | --- |
| genc_id | <6 |
| institution_from_mns | 689000 |
| institution_to_mns | 174000 |
| institution_to_type_mns | 196000 |
| institution_from_type_mns | 727000 |

### `lookup_transfusion_concept` (rows: 0)

| column | null count |
| --- | --- |
| concept_id | <6 |
| vocabulary_id | <6 |
| concept_desc | <6 |

### `lookup_vitals_concept` (rows: 0)

| column | null count |
| --- | --- |
| concept_id | <6 |
| vocabulary_id | <6 |
| concept_desc | <6 |

### `pharmacy_subset` (rows: 84266000)

*Skipped per-column null-fraction check -- too large (>= 5,000,000 rows).*

### `physicians_subset` (rows: 2267000)

| column | null count |
| --- | --- |
| genc_id | <6 |
| admitting_physician_gim | 1601000 |
| discharging_physician_gim | 1544000 |
| mrp_gim | 1500000 |
| mrp_cpso_hashed | 36000 |
| adm_phy_cpso_hashed | 244000 |
| dis_phy_cpso_hashed | 221000 |
| row_num | <6 |
| hospital_num | <6 |

### `radiology_subset` (rows: 8687000)

*Skipped per-column null-fraction check -- too large (>= 5,000,000 rows).*

### `rxnorm_cache` (rows: 713000)

| column | null count |
| --- | --- |
| rxcui | 90000 |
| score | 90000 |
| search_type | <6 |
| raw_input | <6 |
| manually_added | <6 |
| date_added | <6 |
| active | <6 |
| row_num | <6 |

### `vitals_subset` (rows: 411962000)

*Skipped per-column null-fraction check -- too large (>= 5,000,000 rows).*

## Design-critical queries

### Lab concept frequencies (`lab_subset.test_type_mapped_omop`)

| code | concept_desc | n |
| --- | --- | --- |
| 3019550 | nan | 17224000 |
| 3019550 | Sodium [Moles/volume] in Serum or Plasma | 17224000 |
| 3023103 | Potassium [Moles/volume] in Serum or Plasma | 17191000 |
| 3014576 | Chloride [Moles/volume] in Serum or Plasma | 17051000 |
| 3002385 | Erythrocyte distribution width [Ratio] | 16578000 |
| 3000963 | nan | 15603000 |
| 3000963 | Hemoglobin [Mass/volume] in Blood | 15603000 |
| 3010813 | Leukocytes [Presence] in Urine | 15390000 |
| 3010813 | Leukocytes [#/volume] in Blood | 15390000 |
| 3040151 | Glucose [Moles/volume] in Capillary blood | 15196000 |
| 3009542 | Hematocrit [Volume Fraction] of Blood | 15166000 |
| 3007461 | Platelets [#/volume] in Blood | 15056000 |
| 3020564 | Creatinine [Moles/volume] in Serum or Plasma | 14695000 |
| 3024731 | MCV [Entitic volume] | 14693000 |
| 3026361 | Erythrocytes [Presence] in Urine | 13836000 |
| 3026361 | Erythrocytes [#/volume] in Blood | 13836000 |
| 3045716 | Anion gap in Serum or Plasma | 13738000 |
| 3016293 | Bicarbonate [Moles/volume] in Serum or Plasma | 13517000 |
| 3019198 | Lymphocytes [#/volume] in Blood | 13277000 |
| 3017732 | Neutrophils [#/volume] in Blood | 13248000 |
| 3003338 | MCHC [Mass/volume] | 13214000 |
| 3001604 | Monocytes/100 leukocytes in Blood | 12550000 |
| 3001604 | Monocytes [#/volume] in Blood | 12550000 |
| 3006315 | Basophils [#/volume] in Blood | 11991000 |
| 3035941 | MCH [Entitic mass] | 11906000 |
| 3013115 | Eosinophils [#/volume] in Blood | 11901000 |
| 3001123 | Platelet mean volume [Entitic volume] in Blood | 10374000 |
| 3040168 | Immature granulocytes [#/volume] in Blood | 9627000 |
| 40771922 | Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood | 8577000 |
| 3001490 | Nucleated erythrocytes [#/volume] in Blood | 8399000 |
| 3013826 | Glucose [Moles/volume] in Serum or Plasma | 8122000 |
| 3024641 | Urea nitrogen [Moles/volume] in Serum or Plasma | 7569000 |
| 3028615 | Eosinophils [#/volume] in Blood | 7364000 |
| 3028615 | Eosinophils [#/volume] in Blood by Automated count | 7364000 |
| 3033575 | Monocytes [#/volume] in Blood by Automated count | 7361000 |
| 3013429 | Basophils [#/volume] in Blood | 7313000 |
| 3013429 | Basophils [#/volume] in Blood by Automated count | 7313000 |
| 3015377 | Calcium [Moles/volume] in Serum or Plasma | 6848000 |
| 3012095 | Magnesium [Moles/volume] in Serum or Plasma | 6757000 |
| 3012095 | Oxygen saturation [Pure mass fraction] in Venous blood | 6757000 |
| 3003458 | Phosphate [Moles/volume] in Serum or Plasma | 6386000 |
| 3052851 | Hemolysis index of Serum or Plasma | 6139000 |
| 3052851 | Hemolysis interference index of Serum or Plasma | 6139000 |
| 3051611 | Lipemic interference index of Serum or Plasma | 6069000 |
| 3051611 | Lipemic index of Serum or Plasma | 6069000 |
| 3004327 | Lymphocytes [#/volume] in Blood by Automated count | 5741000 |
| 3004327 | Lymphocytes/100 leukocytes in Blood | 5741000 |
| 3004327 | Lymphocytes/100 leukocytes in Blood by Automated count | 5741000 |
| 3024561 | Albumin [Mass/volume] in Serum or Plasma | 5617000 |
| 3013650 | Neutrophils [#/volume] in Blood by Automated count | 5371000 |
| 3013650 | Neutrophils/100 leukocytes in Blood | 5371000 |
| 3051651 | Icteric index of Serum or Plasma | 4855000 |
| 3051651 | Icteric interference index of Serum or Plasma | 4855000 |
| 3032080 | INR in Blood by Coagulation assay | 4402000 |
| 3006140 | Bilirubin.total [Moles/volume] in Serum or Plasma | 3928000 |
| 3006923 | Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma | 3795000 |
| 3035995 | Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma | 3687000 |
| 3053331 | Differential cell count method - Blood | 3523000 |
| 3013466 | aPTT in Blood by Coagulation assay | 3409000 |
| 3034426 | Prothrombin time (PT) | 3100000 |
| 3021447 | Carbon dioxide [Partial pressure] in Venous blood | 2904000 |
| 3045172 | Nucleated erythrocytes [Presence] in Blood by Automated count | 2874000 |
| 3013721 | Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma | 2564000 |
| 3012544 | pH of Venous blood | 2472000 |
| 3002032 | Base excess in Venous blood by calculation | 2310000 |
| 3027801 | Oxygen [Partial pressure] in Arterial blood | 2195000 |
| 3027801 | Oxygen [Partial pressure] in Aterial blood | 2195000 |
| 3027694 | Calcium.ionized [Mass/volume] in Blood | 2144000 |
| 3027694 | Calcium.ionized [Mass/volume] in Serum or Plasma | 2144000 |
| 3027946 | Carbon dioxide [Partial pressure] in Arterial blood | 2131000 |
| 3003396 | Base excess in Arterial blood by calculation | 2042000 |
| 3019977 | pH of Arterial blood | 2029000 |
| 42869600 | Oxygen saturation [Pure mass fraction] in Venous blood | 2014000 |
| 3018010 | Neutrophils/100 leukocytes in Blood | 1996000 |
| 40761511 | CBC panel - Blood by Automated count | 1935000 |
| 3024354 | Oxygen [Partial pressure] in Venous blood | 1850000 |
| 3027995 | Electrolytes 1998 panel - Serum or Plasma | 1778000 |
| 3016502 | Oxygen saturation in Arterial blood | 1739000 |
| 40761509 | Erythrocyte morphology panel - Blood | 1718000 |
| 3027273 | Bicarbonate [Moles/volume] in Venous blood | 1698000 |
| 36306105 | Troponin I.cardiac [Mass/volume] in Serum or Plasma by High sensitivity method | 1667000 |
| 3044916 | Immature granulocytes [Presence] in Blood by Automated count | 1508000 |
| 3037121 | Protein [Mass/volume] in Urine | 1418000 |
| 3007220 | Creatine kinase [Enzymatic activity/volume] in Serum or Plasma | 1380000 |
| 3009609 | Carbon dioxide, total [Moles/volume] in Arterial blood | 1361000 |
| 3009609 | Carbon dioxide, total [Moles/volume] in Venous blood | 1361000 |
| 3028638 | Bilirubin.direct [Moles/volume] in Serum or Plasma | 1338000 |
| 40762528 | nan | 1318000 |
| 3034708 | Nucleated erythrocytes/100 leukocytes [Ratio] in Blood | 1314000 |
| 3008152 | Bicarbonate [Moles/volume] in Arterial blood | 1280000 |
| 3018405 | Lactate [Moles/volume] in Arterial blood | 1274000 |
| 3003137 | Variant lymphocytes [#/volume] in Blood | 1225000 |
| 3008037 | Lactate [Moles/volume] in Venous blood | 1223000 |
| 3024507 | Metamyelocytes [#/volume] in Blood | 1133000 |
| 3021120 | Myelocytes [#/volume] in Blood | 1120000 |
| 3002030 | Lymphocytes/100 leukocytes in Blood | 1105000 |
| 3006504 | Eosinophils/100 leukocytes in Blood | 1099000 |
| 3033543 | Specific gravity of Urine | 1095000 |
| 3015736 | pH of Urine | 1095000 |
| 3045424 | Erythrocytes [Presence] in Urine | 1053000 |
| 3045414 | Leukocytes [Presence] in Urine | 1037000 |
| 3021337 | Troponin I.cardiac [Mass/volume] in Serum or Plasma | 1036000 |
| 3028893 | Ketones [Presence] in Urine | 1035000 |
| 3004905 | Lipase [Enzymatic activity/volume] in Serum or Plasma | 1015000 |
| 3034118 | Platelets Large [Presence] in Blood by Light microscopy | 1013000 |
| 3023081 | Carboxyhemoglobin/Hemoglobin.total in Blood | 1001000 |
| 1003087 | Body temperature | Blood | Specific body temperature | 978000 |
| 3042812 | Nitrite [Presence] in Urine | 973000 |
| 3041084 | Immature granulocytes [#/volume] in Blood by Automated count | 911000 |
| 3034639 | Hemoglobin A1c [Mass/volume] in Blood | 904000 |
| 40760140 | CBC W Auto Differential panel - Blood | 880000 |
| 3022709 | Promyelocytes [#/volume] in Blood | 873000 |
| 3019069 | Monocytes/100 leukocytes in Blood | 845000 |
| 3013869 | Basophils/100 leukocytes in Blood by Automated count | 834000 |
| 3011987 | Polychromasia [Presence] in Blood by Light microscopy | 828000 |
| 3051227 | Blood [Presence] in Urine by Visual | 813000 |
| 3001362 | Plasma cells [#/volume] in Blood | 813000 |
| 3020650 | Glucose [Presence] in Urine | 800000 |
| 3001552 | Prolymphocytes [#/volume] in Blood | 781000 |
| 3026904 | Basophilic stippling [Presence] in Blood by Light microscopy | 781000 |
| 3020460 | C reactive protein [Mass/volume] in Serum or Plasma | 773000 |
| 3020138 | Lactate [Moles/volume] in Arterial blood | 764000 |
| 3020138 | Lactate [Mass/volume] in Serum or Plasma | 764000 |
| 40769783 | Troponin T.cardiac [Mass/volume] in Serum or Plasma by High sensitivity method | 721000 |
| 3025616 | Target cells [Presence] in Blood by Light microscopy | 718000 |
| 3030494 | Manual differential performed [Presence] in Blood | 697000 |
| 3029880 | Horowitz index in Blood | 688000 |
| 3009201 | Thyrotropin [Units/volume] in Serum or Plasma | 677000 |
| 3026910 | Gamma glutamyl transferase [Enzymatic activity/volume] in Serum or Plasma | 672000 |
| 3018199 | Band form neutrophils [#/volume] in Blood | 660000 |
| 3025639 | Microcytes [Presence] in Blood | 656000 |
| 3015183 | Erythrocyte sedimentation rate | 655000 |
| 3029943 | nan | 653000 |
| 3000493 | Elliptocytes [Presence] in Blood by Light microscopy | 637000 |
| 3005854 | Burr cells [Presence] in Blood by Light microscopy | 633000 |
| 3047107 | Calcium [Mass/volume] corrected for albumin in Serum or Plasma | 619000 |
| 3050169 | Erythrocyte inclusion bodies [Identifier] in Blood | 612000 |
| 3023368 | Bacteria identified in Blood by Culture | 602000 |
| 3021502 | Macrocytes [Presence] in Blood by Light microscopy | 594000 |
| 3031042 | Nucleated cells [#/volume] in Blood | 586000 |
| 3015632 | Carbon dioxide, total [Moles/volume] in Serum or Plasma | 556000 |
| 3019416 | Acanthocytes [Presence] in Blood by Light microscopy | 544000 |
| 3011633 | Myeloblasts [#/volume] in Blood by Manual count | 535000 |
| 3035529 | Creatinine renal clearance predicted by Cockcroft-Gault formula | 523000 |
| 3005481 | Spherocytes [Presence] in Blood by Light microscopy | 521000 |
| 3024783 | Stomatocytes [Presence] in Blood by Light microscopy | 511000 |
| 3034824 | Hypochromia [Presence] in Blood | 510000 |
| 3029414 | Hydrogen ion [Moles/volume] in Venous blood | 506000 |
| 3036273 | Rouleaux [Presence] in Blood by Light microscopy | 506000 |
| 3010950 | Oval macrocytes [Presence] in Blood by Light microscopy | 504000 |
| 3016771 | Amylase [Enzymatic activity/volume] in Serum or Plasma | 502000 |
| 3002620 | Howell-Jolly bodies [Presence] in Blood by Light microscopy | 501000 |
| 3019761 | Pappenheimer bodies [Presence] in Blood by Light microscopy | 500000 |
| 36661377 | SARS-CoV-2 (COVID-19) RNA [Presence] in Respiratory specimen by Sequencing | 496000 |
| 3035173 | Hydrogen ion [Moles/volume] in Arterial blood | 492000 |
| 3024889 | Methemoglobin/Hemoglobin.total in Venous blood | 492000 |
| 3023089 | Hairy cells [#/volume] in Blood | 490000 |
| 3034022 | Lactate dehydrogenase panel - Serum or Plasma | 489000 |
| 3020412 | Sickle cells [Presence] in Blood by Light microscopy | 484000 |
| 36305091 | Oxyhemoglobin [Mass/volume] in Plasma | 481000 |
| 40761551 | Bilirubin [Presence] in Urine by Confirmatory method | 479000 |
| 3000456 | Dacrocytes [Presence] in Blood by Light microscopy | 475000 |
| 3026805 | Blister cells [Presence] in Blood by Light microscopy | 474000 |
| 3019900 | Cholesterol [Moles/volume] in Serum or Plasma | 462000 |
| 3007930 | Methemoglobin/Hemoglobin.total in Blood | 457000 |
| 3025839 | Triglyceride [Moles/volume] in Serum or Plasma | 455000 |
| 3053213 | nan | 451000 |
| 3009035 | Cobalamin (Vitamin B12) [Moles/volume] in Serum or Plasma | 443000 |
| 3010517 | Carboxyhemoglobin/Hemoglobin.total in Venous blood | 438000 |
| 3001657 | Leukocytes [#/volume] corrected for nucleated erythrocytes in Blood | 427000 |
| 3023520 | Reticulocytes [#/volume] in Blood | 411000 |
| 3003214 | Platelet morphology finding [Identifier] in Blood | 406000 |
| 3023602 | Cholesterol in HDL [Moles/volume] in Serum or Plasma | 400000 |
| 3047181 | nan | 400000 |
| 1175793 | Methicillin resistant Staphylococcus aureus (MRSA) DNA [Presence] in Nose by NAA with probe detection | 396000 |
| 3001122 | Ferritin [Mass/volume] in Serum or Plasma | 385000 |
| 3048233 | Platelet aggregation panel - Platelet rich plasma | 383000 |
| 37393011 | nan | 383000 |
| 3007876 | Appearance of Urine | 382000 |
| 3029396 | Erythrocyte agglutination [Presence] in Blood | 378000 |
| 3015586 | Segmented neutrophils [#/volume] in Blood | 375000 |
| 3011960 | Natriuretic peptide B [Mass/volume] in Serum or Plasma | 370000 |
| 3001308 | Cholesterol in LDL [Moles/volume] in Serum or Plasma | 356000 |
| 3008295 | Osmolality of Serum or Plasma | 352000 |
| 3018251 | Fasting glucose [Moles/volume] in Serum or Plasma | 346000 |
| 3005147 | Methemoglobin [Mass/volume] in Blood | 341000 |
| 40760142 | nan | 334000 |
| 3007242 | Bilirubin.indirect [Moles/volume] in Serum or Plasma | 331000 |
| 3031741 | Smudge cells [#/volume] in Blood | 329000 |
| 3002937 | Fibrinogen [Presence] in Platelet poor plasma | 329000 |
| 3031750 | Smear morphology panel - Blood | 325000 |
| 46235370 | Lactate dehydrogenase in body fluid/Lactate dehydrogenase in serum | 316000 |
| 3032039 | Urobilinogen [Moles/volume] in Urine | 313000 |
| 4022146 | Leukocyte count corrected for nucleated erythrocytes | 312000 |
| 40763580 | Pseudo Pelger Huet cells [Presence] in Blood by Light microscopy | 309000 |
| 3004789 | Transferrin [Mass/volume] in Serum or Plasma | 307000 |
| 3019880 | Schistocytes [Presence] in Blood by Light microscopy | 306000 |
| 3009577 | Iron binding capacity [Moles/volume] in Serum or Plasma | 305000 |
| 3016038 | Potassium [Moles/volume] in Urine | 301000 |
| 3003181 | Sodium [Moles/volume] in Urine | 297000 |

### Vitals concept frequencies (`vitals_subset.measurement_mapped_omop`)

| code | concept_desc | n |
| --- | --- | --- |
| 3013502 | Oxygen saturation in Blood | 60564000 |
| 3027018 | Heart rate | 55511000 |
| 3024171 | Respiratory rate | 50343000 |
| 3020891 | Body temperature | 39124000 |
| 36203185 | Blood pressure panel with all children optional | 25627000 |
| 3004249 | Systolic blood pressure | 22549000 |
| 3012888 | Diastolic blood pressure | 21377000 |
| 3034263 | Pain severity - Reported | 6417000 |
| 3014080 | Oxygen gas flow Oxygen delivery system | 5078000 |
| 3005629 | Inhaled oxygen flow rate | 4526000 |
| 3020716 | Inhaled oxygen concentration | 2622000 |
| 3025315 | Body weight | 1498000 |
| 4326744 | Blood pressure | 1046000 |

### Lab unit samples, top ~30 concepts (`lab_subset.result_unit`)

| code | unit | n |
| --- | --- | --- |
| 3000963 | g/L | 14352000 |
| 3000963 | G/L | 969000 |
| 3000963 |  | 266000 |
| 3000963 | G/DL | 9000 |
| 3000963 | None | 6000 |
| 3000963 | (null) | 1000 |
| 3001123 | fL | 10203000 |
| 3001123 |  | 164000 |
| 3001123 | None | 6000 |
| 3001123 | (null) | 1000 |
| 3001123 | fl | 0 |
| 3001490 | X10 9/L | 1847000 |
| 3001490 | % | 1518000 |
| 3001490 | /100 LKC | 1207000 |
| 3001490 | /100 WBC | 788000 |
| 3001490 | /100(WBCs) | 373000 |
| 3001490 | X10  9/L | 333000 |
| 3001490 | /100WBC | 321000 |
| 3001490 | X 10 9/L | 315000 |
| 3001490 | x 10^9/L | 314000 |
| 3001490 | X 10^9/L | 311000 |
| 3001490 | x10 9/L | 233000 |
| 3001490 | /100LKC | 232000 |
| 3001490 | x10^9/L | 209000 |
| 3001490 | NULL | 207000 |
| 3001490 | /100 WBC's | 87000 |
| 3001490 |  | 66000 |
| 3001490 | /100 WBCs | 25000 |
| 3001490 | X10^9/L | 14000 |
| 3001490 | /TCC | 0 |
| 3001604 | % | 2001000 |
| 3001604 | X10 9/L | 1492000 |
| 3001604 | NULL | 1375000 |
| 3001604 | x10E9/L | 968000 |
| 3001604 | x10 9/L | 876000 |
| 3001604 | E9/L | 720000 |
| 3001604 | x10^9/L | 706000 |
| 3001604 |  | 634000 |
| 3001604 | x10e9/L | 626000 |
| 3001604 | 10*9/L | 568000 |
| 3001604 | 10e9/L | 543000 |
| 3001604 | 10E9/L | 459000 |
| 3001604 | X10  9/L | 373000 |
| 3001604 | x 10^9/L | 336000 |
| 3001604 | X 10 9/L | 317000 |
| 3001604 | X 10^9/L | 312000 |
| 3001604 | RATIO | 232000 |
| 3001604 | (null) | 5000 |
| 3001604 | x10E6/L | 3000 |
| 3001604 | None | 2000 |
| 3001604 | X10^9/L | 1000 |
| 3001604 | ratio | 0 |
| 3002385 | % | 12130000 |
| 3002385 | fL | 2014000 |
| 3002385 |  | 858000 |
| 3002385 | %CV | 626000 |
| 3002385 | CV | 527000 |
| 3002385 | % cv | 417000 |
| 3002385 | None | 6000 |
| 3002385 | (null) | 1000 |
| 3003338 | g/L | 12503000 |
| 3003338 |  | 704000 |
| 3003338 | None | 6000 |
| 3003338 | (null) | 1000 |
| 3006140 | umol/L | 3558000 |
| 3006140 | UMOL/L | 326000 |
| 3006140 |  | 42000 |
| 3006140 | None | 1000 |
| 3006140 | (null) | 0 |
| 3006315 | % | 2292000 |
| 3006315 | X10 9/L | 1466000 |
| 3006315 | NULL | 1294000 |
| 3006315 | x10E9/L | 938000 |
| 3006315 | E9/L | 720000 |
| 3006315 | x10 9/L | 718000 |
| 3006315 | x10^9/L | 704000 |
| 3006315 | x10e9/L | 626000 |
| 3006315 | 10e9/L | 525000 |
| 3006315 | 10*9/L | 522000 |
| 3006315 | 10E9/L | 459000 |
| 3006315 | X10  9/L | 332000 |
| 3006315 |  | 327000 |
| 3006315 | X 10 9/L | 317000 |
| 3006315 | X 10^9/L | 312000 |
| 3006315 | RATIO | 232000 |
| 3006315 | x 10^9/L | 187000 |
| 3006315 | (null) | 20000 |
| 3006315 | None | 1000 |
| 3006315 | x10E6/L | 0 |
| 3006315 | ratio | 0 |
| 3007461 | x10^9/L | 2335000 |
| 3007461 | x10 9/L | 2329000 |
| 3007461 | X10 9/L | 2158000 |
| 3007461 | x10E9/L | 1362000 |
| 3007461 | x10*9/L | 1028000 |
| 3007461 |  | 981000 |
| 3007461 | x 10^9/L | 838000 |
| 3007461 | E9/L | 712000 |
| 3007461 | X10^9/L | 643000 |
| 3007461 | x10e9/L | 626000 |
| 3007461 | 10*9/L | 549000 |
| 3007461 | 10e9/L | 532000 |
| 3007461 | 10E9/L | 469000 |
| 3007461 | X 10^9/L | 310000 |
| 3007461 | NULL | 174000 |
| 3007461 | None | 7000 |
| 3007461 | (null) | 1000 |
| 3007461 | 10^9/L | 1000 |
| 3007461 | umol/L | 1000 |
| 3007461 | /L | 0 |
| 3007461 | x E9/L | 0 |
| 3008037 | mmol/L | 1211000 |
| 3008037 |  | 12000 |
| 3009542 | L/L | 12093000 |
| 3009542 | NULL | 1315000 |
| 3009542 | L | 968000 |
| 3009542 |  | 497000 |
| 3009542 | % | 282000 |
| 3009542 | None | 6000 |
| 3009542 | mmol/L | 3000 |
| 3009542 | (null) | 1000 |
| 3010813 | x10 9/L | 2759000 |
| 3010813 | x10^9/L | 2590000 |
| 3010813 | X10 9/L | 2164000 |
| 3010813 | x10E9/L | 1400000 |
| 3010813 | x10*9/L | 1019000 |
| 3010813 | E9/L | 752000 |
| 3010813 |  | 691000 |
| 3010813 | X10^9/L | 654000 |
| 3010813 | x10e9/L | 626000 |
| 3010813 | x 10^9/L | 591000 |
| 3010813 | X 10^9/L | 557000 |
| 3010813 | 10*9/L | 550000 |
| 3010813 | 10e9/L | 533000 |
| 3010813 | 10E9/L | 469000 |
| 3010813 | /hpf | 22000 |
| 3010813 | None | 7000 |
| 3010813 | cells/uL | 2000 |
| 3010813 | X 10^6/L | 1000 |
| 3010813 | x 10^6/L | 1000 |
| 3010813 | (null) | 1000 |
| 3010813 | x10E6/L | 0 |
| 3010813 | x E9/L | 0 |
| 3010813 | /HPF | <6 |
| 3013115 | % | 1978000 |
| 3013115 | X10 9/L | 1466000 |
| 3013115 | NULL | 1349000 |
| 3013115 | x10E9/L | 980000 |
| 3013115 | x10 9/L | 731000 |
| 3013115 | E9/L | 720000 |
| 3013115 | x10^9/L | 705000 |
| 3013115 | x10e9/L | 626000 |
| 3013115 | 10e9/L | 531000 |
| 3013115 | 10*9/L | 524000 |
| 3013115 | 10E9/L | 459000 |
| 3013115 |  | 363000 |
| 3013115 | X10  9/L | 342000 |
| 3013115 | X 10 9/L | 317000 |
| 3013115 | X 10^9/L | 312000 |
| 3013115 | x 10^9/L | 249000 |
| 3013115 | RATIO | 232000 |
| 3013115 | (null) | 16000 |
| 3013115 | None | 2000 |
| 3013115 | x10E6/L | 1000 |
| 3013115 | ratio | 0 |
| 3013115 | X 10^6/L | 0 |
| 3013826 | mmol/L | 7496000 |
| 3013826 | MMOL/L | 454000 |
| 3013826 |  | 146000 |
| 3013826 | % | 22000 |
| 3013826 | None | 2000 |
| 3013826 | (null) | 1000 |
| 3013826 | g/L | <6 |
| 3014576 | mmol/L | 15892000 |
| 3014576 | MMOL/L | 987000 |
| 3014576 |  | 168000 |
| 3014576 | None | 4000 |
| 3014576 | (null) | 1000 |
| 3016293 | mmol/L | 13114000 |
| 3016293 | MMOL/L | 198000 |
| 3016293 |  | 168000 |
| 3016293 | mmoL/L | 28000 |
| 3016293 | None | 7000 |
| 3016293 | (null) | 2000 |
| 3016293 | % | <6 |
| 3017732 | % | 2014000 |
| 3017732 | X10 9/L | 1482000 |
| 3017732 | x10^9/L | 1359000 |
| 3017732 | NULL | 1263000 |
| 3017732 | x10E9/L | 977000 |
| 3017732 | x 10^9/L | 792000 |
| 3017732 | E9/L | 720000 |
| 3017732 | x10 9/L | 673000 |
| 3017732 | x10e9/L | 626000 |
| 3017732 | 10*9/L | 572000 |
| 3017732 | 10e9/L | 545000 |
| 3017732 |  | 477000 |
| 3017732 | 10E9/L | 459000 |
| 3017732 | X10  9/L | 376000 |
| 3017732 | X 10^9/L | 312000 |
| 3017732 | X 10 9/L | 296000 |
| 3017732 | RATIO | 232000 |
| 3017732 | (null) | 43000 |
| 3017732 | x10E6/L | 9000 |
| 3017732 | x10*9/L | 8000 |
| 3017732 | None | 8000 |
| 3017732 | x10 6/L | 3000 |
| 3017732 | X10^9/L | 1000 |
| 3017732 | X 10^6/L | 0 |
| 3018405 | mmol/L | 1231000 |
| 3018405 |  | 39000 |
| 3018405 | None | 4000 |
| 3018405 | (null) | 0 |
| 3019198 | % | 2042000 |
| 3019198 | X10 9/L | 1932000 |
| 3019198 | NULL | 1391000 |
| 3019198 | x10^9/L | 1235000 |
| 3019198 | x10E9/L | 972000 |
| 3019198 | x 10^9/L | 777000 |
| 3019198 | E9/L | 720000 |
| 3019198 | x10e9/L | 626000 |
| 3019198 | 10*9/L | 571000 |
| 3019198 | 10e9/L | 544000 |
| 3019198 | 10E9/L | 459000 |
| 3019198 |  | 447000 |
| 3019198 | X10  9/L | 377000 |
| 3019198 | X 10 9/L | 316000 |
| 3019198 | X 10^9/L | 312000 |
| 3019198 | x10 9/L | 291000 |
| 3019198 | RATIO | 232000 |
| 3019198 | x10E6/L | 9000 |
| 3019198 | x10*9/L | 8000 |
| 3019198 | ratio | 6000 |
| 3019198 | X10  6/L | 6000 |
| 3019198 | (null) | 2000 |
| 3019198 | None | 2000 |
| 3019198 | X10^9/L | 1000 |
| 3019198 | X 10^6/L | 0 |
| 3019550 | mmol/L | 16031000 |
| 3019550 | MMOL/L | 988000 |
| 3019550 |  | 199000 |
| 3019550 | None | 4000 |
| 3019550 | (null) | 1000 |
| 3020138 | mmol/L | 764000 |
| 3020138 |  | <6 |
| 3020564 | umol/L | 12964000 |
| 3020564 | UMOL/L | 1383000 |
| 3020564 |  | 206000 |
| 3020564 | Umol/L | 137000 |
| 3020564 | None | 4000 |
| 3020564 | (null) | 1000 |
| 3020564 | mmol/L | 0 |
| 3020564 | MMOL/D | 0 |
| 3020564 | MMOL/L | 0 |
| 3020564 | mmol/d | 0 |
| 3023103 | mmol/L | 15965000 |
| 3023103 | MMOL/L | 1016000 |
| 3023103 |  | 201000 |
| 3023103 | None | 8000 |
| 3023103 | (null) | 1000 |
| 3024641 | mmol/L | 6886000 |
| 3024641 | MMOL/L | 655000 |
| 3024641 |  | 28000 |
| 3024641 | None | 0 |
| 3024641 | (null) | 0 |
| 3024731 | fL | 11854000 |
| 3024731 | fl | 1178000 |
| 3024731 | FL | 968000 |
| 3024731 | pg | 532000 |
| 3024731 |  | 153000 |
| 3024731 | None | 6000 |
| 3024731 | (null) | 1000 |
| 3026361 | x10^12/L | 2707000 |
| 3026361 | x10 12/L | 2324000 |
| 3026361 | X10 12/L | 2186000 |
| 3026361 | x10E12/L | 1094000 |
| 3026361 | x10*12/L | 1031000 |
| 3026361 | E12/L | 712000 |
| 3026361 | X10^12/L | 649000 |
| 3026361 | x10e12/L | 641000 |
| 3026361 | 10*12/L | 550000 |
| 3026361 | 10e12/L | 532000 |
| 3026361 |  | 485000 |
| 3026361 | x 10^12/L | 469000 |
| 3026361 | X 10^12/L | 311000 |
| 3026361 | NULL | 109000 |
| 3026361 | /hpf | 22000 |
| 3026361 | None | 6000 |
| 3026361 | X 10^6/L | 4000 |
| 3026361 | x10 6/L | 2000 |
| 3026361 | x 10^6/L | 1000 |
| 3026361 | (null) | 1000 |
| 3026361 | x10E6/L | 0 |
| 3026361 | x E12/L | 0 |
| 3026361 | /HPF | <6 |
| 3032080 |  | 3131000 |
| 3032080 | INR | 558000 |
| 3032080 | NULL | 521000 |
| 3032080 | None | 156000 |
| 3032080 | (null) | 36000 |
| 3035941 | pg | 10778000 |
| 3035941 | PG | 968000 |
| 3035941 |  | 153000 |
| 3035941 | None | 6000 |
| 3035941 | (null) | 1000 |
| 3040151 | mmol/L | 12935000 |
| 3040151 | MMOL/L | 1259000 |
| 3040151 | None | 477000 |
| 3040151 |  | 417000 |
| 3040151 | (null) | 108000 |
| 3040168 | X10 9/L | 1778000 |
| 3040168 | % | 1678000 |
| 3040168 | NULL | 1040000 |
| 3040168 | x10*9/L | 942000 |
| 3040168 | E9/L | 627000 |
| 3040168 | x10 9/L | 607000 |
| 3040168 | RATIO | 548000 |
| 3040168 | x10e9/L | 430000 |
| 3040168 | x10^9/L | 421000 |
| 3040168 | 10*9/L | 420000 |
| 3040168 | X 10 9/L | 317000 |
| 3040168 | X 10^9/L | 300000 |
| 3040168 | x10E9/L | 218000 |
| 3040168 | x 10^9/L | 159000 |
| 3040168 |  | 116000 |
| 3040168 | None | 24000 |
| 3045716 | mmol/L | 10578000 |
| 3045716 |  | 1630000 |
| 3045716 | MMOL/L | 914000 |
| 3045716 | None | 328000 |
| 3045716 | (null) | 288000 |
| 3045716 | mmol/d | 1000 |
| 40771922 | mL/min/1.73m2 | 2836000 |
| 40771922 |  | 1541000 |
| 40771922 | NULL | 939000 |
| 40771922 | ML/MIN | 892000 |
| 40771922 | mL/min/1.73m*2 | 569000 |
| 40771922 | mL/min/1.73m^2 | 497000 |
| 40771922 | mL/min/1.73 m2 | 391000 |
| 40771922 | * | 373000 |
| 40771922 | mL/min/1.73m� | 277000 |
| 40771922 | mL/min/1.73mÂ² | 122000 |
| 40771922 | ml/min | 67000 |
| 40771922 | mL/min | 58000 |
| 40771922 | mL/min/1.73m² | 12000 |
| 40771922 | mL/min/1.7 | 2000 |
| 40771922 | None | 1000 |
| 40771922 | (null) | 0 |

### Vitals unit samples, all 13 concepts (`vitals_subset.measurement_unit`)

| code | unit | n |
| --- | --- | --- |
| 3004249 | None | 8718000 |
| 3004249 | mmHg | 7145000 |
| 3004249 | mmHd | 3556000 |
| 3004249 |  | 3128000 |
| 3004249 | mm Hg | 2000 |
| 3004249 | NULL | 0 |
| 3005629 |  | 2589000 |
| 3005629 | None | 1937000 |
| 3005629 | corked | 0 |
| 3005629 | 5 | 0 |
| 3005629 | RA | 0 |
| 3005629 | air | 0 |
| 3005629 | room air | 0 |
| 3005629 | on ear | 0 |
| 3005629 | Corked | 0 |
| 3005629 | with speaking valve | 0 |
| 3005629 | with cork | 0 |
| 3005629 | cpap | <6 |
| 3005629 | 5L | <6 |
| 3005629 | CPAP | <6 |
| 3005629 | AIR | <6 |
| 3005629 | corked. | <6 |
| 3005629 | trach corked | <6 |
| 3005629 | A | <6 |
| 3005629 | lying on R side | <6 |
| 3005629 | decannulated | <6 |
| 3005629 | Room Air | <6 |
| 3005629 | L/min | <6 |
| 3005629 | Standing weight | <6 |
| 3005629 | 1.5L | <6 |
| 3005629 | 3L | <6 |
| 3005629 | optiflow | <6 |
| 3005629 | open trach. | <6 |
| 3005629 | on FINGERS | <6 |
| 3005629 | Started on 2L NP | <6 |
| 3005629 | W speaking valve | <6 |
| 3005629 | a | <6 |
| 3005629 | bet weight | <6 |
| 3005629 | cork. | <6 |
| 3005629 | inh | <6 |
| 3005629 | min | <6 |
| 3005629 | nebs | <6 |
| 3005629 | nebulizer | <6 |
| 3005629 | sleeping | <6 |
| 3005629 | speaking valve removed | <6 |
| 3005629 | thus 4L reapplied now 95% | <6 |
| 3005629 | uncorked | <6 |
| 3005629 | uncorked trach | <6 |
| 3005629 | ventolin nebulizer | <6 |
| 3005629 | with DB | <6 |
| 3005629 | taking off optiflow | <6 |
| 3005629 | needs DBC exercises to inc | <6 |
| 3005629 | on RA | <6 |
| 3005629 | on fingers | <6 |
| 3005629 | post walk in the hallway | <6 |
| 3005629 | pt dislikes np | <6 |
| 3005629 | pt removed NP | <6 |
| 3005629 | refused oxygen | <6 |
| 3005629 | refusing oxygen | <6 |
| 3005629 | -air | <6 |
| 3005629 | 0.5 L | <6 |
| 3005629 | 1.5 L | <6 |
| 3005629 | 1.5 liters | <6 |
| 3005629 | 10 cmH2O; RA | <6 |
| 3005629 | 12 cmH2O + 4 lpm | <6 |
| 3005629 | 12 cmH2O + 4lpm | <6 |
| 3005629 | 14/7 | <6 |
| 3005629 | 4L | <6 |
| 3005629 | 60L | <6 |
| 3005629 | 93% in chair | <6 |
| 3005629 | Airvo Nasal Prong | <6 |
| 3005629 | Corked Trach | <6 |
| 3005629 | DB+C | <6 |
| 3005629 | GIVEN 3L O2 VIA NP | <6 |
| 3005629 | HD started | <6 |
| 3005629 | HR elevated to 108 when pt stood up | <6 |
| 3005629 | RA) | <6 |
| 3005629 | Room air | <6 |
| 3012888 | None | 8712000 |
| 3012888 | mmHg | 5989000 |
| 3012888 | mmHd | 3546000 |
| 3012888 |  | 3128000 |
| 3012888 | mm Hg | 2000 |
| 3012888 | NULL | 0 |
| 3013502 | % | 35244000 |
| 3013502 | None | 22125000 |
| 3013502 |  | 3103000 |
| 3013502 | Percentage | 91000 |
| 3013502 | NULL | 0 |
| 3014080 | None | 4728000 |
| 3014080 |  | 350000 |
| 3020716 | None | 2622000 |
| 3020891 | None | 17832000 |
| 3020891 | degrees C | 3515000 |
| 3020891 |  | 3287000 |
| 3020891 | Fahrenheit | 2983000 |
| 3020891 | Cel | 2816000 |
| 3020891 | DegC | 2740000 |
| 3020891 | NULL | 1855000 |
| 3020891 | Celcius | 1726000 |
| 3020891 | degC | 903000 |
| 3020891 | Deg C | 781000 |
| 3020891 | C | 347000 |
| 3020891 | Celsius | 340000 |
| 3024171 | None | 19962000 |
| 3024171 |  | 15476000 |
| 3024171 | br/min | 5164000 |
| 3024171 | breaths/min | 4135000 |
| 3024171 | NULL | 3144000 |
| 3024171 | breaths/minute | 2360000 |
| 3024171 | Breaths Per Minute | 91000 |
| 3024171 | bpm | 12000 |
| 3025315 | None | 1286000 |
| 3025315 |  | 212000 |
| 3025315 | kg | 0 |
| 3025315 | lb | <6 |
| 3027018 | None | 21217000 |
| 3027018 |  | 16902000 |
| 3027018 | bpm | 7223000 |
| 3027018 | beats per minute | 4140000 |
| 3027018 | NULL | 3309000 |
| 3027018 | beats/min | 2366000 |
| 3027018 | BPM | 261000 |
| 3027018 | Beats Per Minute | 93000 |
| 3034263 |  | 4463000 |
| 3034263 | None | 1954000 |
| 4326744 | None | 1046000 |
| 36203185 | None | 10273000 |
| 36203185 |  | 9976000 |
| 36203185 | mmHg | 2963000 |
| 36203185 | NULL | 2324000 |
| 36203185 | Millimeters of Mercury (mmHg) | 91000 |

### Table date ranges (year only)

| table | column | min year | max year |
| --- | --- | --- | --- |
| admdad_subset | admission_date_time | 2010 | 2024 |
| admdad_subset | discharge_date_time | 2015 | 2024 |
| er_subset | triage_date_time | 2012 | 2024 |
| er_subset | disposition_date_time | 2012 | 2024 |
| ipscu_subset | scu_admit_date_time | 2010 | 2024 |
| ipscu_subset | scu_discharge_date_time | 2010 | 2024 |
| lab_subset | collection_date_time | 1945 | 2025 |
| vitals_subset | measure_date_time | 2000 | 2025 |
| pharmacy_subset | med_start_date_time | 1930 | 9022 |
| pharmacy_subset | med_end_date_time | 1840 | 8186 |
| radiology_subset | ordered_date_time | 1979 | 2025 |
| radiology_subset | performed_date_time | 1915 | 9999 |

### Per-hospital data coverage (`lookup_data_coverage`)

| data | min_date | max_date | hospital_num | additional_info |
| --- | --- | --- | --- | --- |
| echo | None | None | 127 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 127 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 127 |  |
| iphig | 2015-04-01 | 2024-06-30 | 127 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 127 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 127 |  |
| er | 2015-04-01 | 2024-06-30 | 127 |  |
| lab | 2020-01-01 | 2024-06-30 | 127 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 127 |  |
| vitals | 2020-01-01 | 2024-06-30 | 127 |  |
| radiology | 2020-01-01 | 2024-06-30 | 127 |  |
| roomtransfer | 2020-01-01 | 2024-06-30 | 127 |  |
| transfusion | 2020-01-03 | 2024-06-30 | 127 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 127 |  |
| physicians | 2015-04-01 | 2024-06-30 | 127 |  |
| pharmacy | 2016-07-20 | 2016-07-20 | 127 |  |
| pharmacy | 2019-09-23 | 2024-06-30 | 127 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 127 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 127 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 127 |  |
| admdad | 2015-04-01 | 2024-06-30 | 127 |  |
| echo | None | None | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| locality_variables | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| ipintervention | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| iphig | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| erdiagnosis | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| erconsults | 2022-04-07 | 2024-06-27 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| er | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| lab | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| derived_variables | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| vitals | None | None | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| radiology | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| roomtransfer | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| transfusion | 2022-04-07 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| ipscu | 2022-04-06 | 2022-10-10 | 132 | Hospital does not an intensive care unit, provides limited step-down care through observation unit |
| ipscu | 2022-11-10 | 2024-06-27 | 132 | Hospital does not an intensive care unit, provides limited step-down care through observation unit |
| physicians | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| pharmacy | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| ipcmg | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| ipdiagnosis | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| erintervention | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| admdad | 2022-04-01 | 2024-06-30 | 132 | Hospital does not have a dedicated intensive care unit, provides limited step-down care through observation unit |
| echo | 2022-05-04 | 2022-05-04 | 133 |  |
| echo | 2022-06-22 | 2022-07-16 | 133 |  |
| echo | 2022-09-23 | 2022-10-18 | 133 |  |
| echo | 2022-12-09 | 2022-12-09 | 133 |  |
| echo | 2023-01-20 | 2023-02-04 | 133 |  |
| echo | 2023-06-26 | 2023-07-25 | 133 |  |
| echo | 2023-09-25 | 2023-09-25 | 133 |  |
| echo | 2024-02-09 | 2024-02-09 | 133 |  |
| echo | 2024-05-24 | 2024-05-24 | 133 |  |
| locality_variables | 2022-04-01 | 2024-06-30 | 133 |  |
| ipintervention | 2022-04-01 | 2024-06-30 | 133 |  |
| iphig | 2022-04-01 | 2024-06-30 | 133 |  |
| erdiagnosis | 2022-04-01 | 2024-06-30 | 133 |  |
| erconsults | 2022-04-01 | 2024-06-29 | 133 |  |
| er | 2022-04-01 | 2024-06-30 | 133 |  |
| lab | 2022-04-01 | 2024-06-30 | 133 |  |
| derived_variables | 2022-04-01 | 2024-06-30 | 133 |  |
| vitals | None | None | 133 |  |
| radiology | 2022-04-01 | 2024-06-30 | 133 |  |
| roomtransfer | 2022-04-01 | 2024-06-30 | 133 |  |
| transfusion | 2022-04-01 | 2024-06-30 | 133 |  |
| ipscu | 2022-04-01 | 2024-06-29 | 133 |  |
| physicians | 2022-04-01 | 2024-06-30 | 133 |  |
| pharmacy | 2022-04-01 | 2024-06-30 | 133 |  |
| ipcmg | 2022-04-01 | 2024-06-30 | 133 |  |
| ipdiagnosis | 2022-04-01 | 2024-06-30 | 133 |  |
| erintervention | 2022-04-01 | 2024-06-30 | 133 |  |
| admdad | 2022-04-01 | 2024-06-30 | 133 |  |
| echo | 2015-04-01 | 2021-03-31 | 128 |  |
| echo | 2021-11-12 | 2023-08-11 | 128 |  |
| echo | 2023-09-28 | 2023-10-10 | 128 |  |
| echo | 2024-04-02 | 2024-05-16 | 128 |  |
| echo | 2024-06-25 | 2024-06-25 | 128 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 128 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 128 |  |
| iphig | 2015-04-01 | 2024-06-30 | 128 |  |
| erdiagnosis | 2015-09-22 | 2024-06-30 | 128 |  |
| erconsults | 2015-09-22 | 2024-06-30 | 128 |  |
| er | 2015-09-22 | 2024-06-30 | 128 |  |
| lab | 2017-02-01 | 2024-06-30 | 128 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 128 |  |
| vitals | None | None | 128 |  |
| radiology | 2015-04-01 | 2024-06-30 | 128 |  |
| roomtransfer | 2015-04-01 | 2021-01-31 | 128 |  |
| roomtransfer | 2021-04-01 | 2024-06-30 | 128 |  |
| transfusion | 2015-04-01 | 2024-06-30 | 128 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 128 |  |
| physicians | 2015-04-01 | 2024-06-30 | 128 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 128 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 128 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 128 |  |
| erintervention | 2021-04-01 | 2021-11-12 | 128 |  |
| erintervention | 2022-04-01 | 2024-06-30 | 128 |  |
| admdad | 2015-04-01 | 2024-06-30 | 128 |  |
| echo | 2019-04-01 | 2024-06-30 | 109 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 109 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 109 |  |
| iphig | 2015-04-01 | 2024-06-30 | 109 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 109 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 109 |  |
| er | 2015-04-01 | 2024-06-30 | 109 |  |
| lab | 2015-04-01 | 2024-06-30 | 109 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 109 |  |
| vitals | None | None | 109 |  |
| radiology | 2015-04-01 | 2024-06-30 | 109 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 109 |  |
| transfusion | None | None | 109 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 109 |  |
| physicians | 2015-04-01 | 2024-06-30 | 109 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 109 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 109 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 109 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 109 |  |
| admdad | 2015-04-01 | 2024-06-30 | 109 |  |
| echo | 2019-04-01 | 2024-06-30 | 110 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 110 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 110 |  |
| iphig | 2015-04-01 | 2024-06-30 | 110 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 110 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 110 |  |
| er | 2015-04-01 | 2024-06-30 | 110 |  |
| lab | 2015-04-01 | 2024-06-30 | 110 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 110 |  |
| vitals | None | None | 110 |  |
| radiology | 2015-04-01 | 2024-06-30 | 110 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 110 |  |
| transfusion | None | None | 110 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 110 |  |
| physicians | 2015-04-01 | 2024-06-30 | 110 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 110 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 110 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 110 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 110 |  |
| admdad | 2015-04-01 | 2024-06-30 | 110 |  |
| echo | 2015-04-01 | 2024-06-30 | 129 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 129 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 129 |  |
| iphig | 2015-04-01 | 2024-06-30 | 129 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 129 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 129 |  |
| er | 2015-04-01 | 2024-06-30 | 129 |  |
| lab | 2015-04-01 | 2024-06-30 | 129 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 129 |  |
| vitals | 2015-04-01 | 2024-06-30 | 129 |  |
| radiology | 2015-04-01 | 2024-06-30 | 129 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 129 |  |
| transfusion | 2015-04-01 | 2024-06-30 | 129 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 129 |  |
| physicians | 2015-04-01 | 2024-06-30 | 129 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 129 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 129 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 129 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 129 |  |
| admdad | 2015-04-01 | 2024-06-30 | 129 |  |
| echo | 2015-04-01 | 2024-06-30 | 130 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 130 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 130 |  |
| iphig | 2015-04-01 | 2024-06-30 | 130 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 130 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 130 |  |
| er | 2015-04-01 | 2024-06-30 | 130 |  |
| lab | 2015-04-01 | 2024-06-28 | 130 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 130 |  |
| vitals | None | None | 130 |  |
| radiology | 2015-04-01 | 2024-06-30 | 130 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 130 |  |
| transfusion | 2015-04-01 | 2024-06-30 | 130 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 130 |  |
| physicians | 2015-04-01 | 2024-06-30 | 130 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 130 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 130 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 130 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 130 |  |
| admdad | 2015-04-01 | 2024-06-30 | 130 |  |
| echo | 2022-07-01 | 2024-06-30 | 117 |  |
| locality_variables | 2018-10-01 | 2024-06-30 | 117 |  |
| ipintervention | 2018-10-01 | 2024-06-30 | 117 |  |
| iphig | 2018-10-01 | 2024-06-30 | 117 |  |
| erdiagnosis | 2018-10-01 | 2024-06-30 | 117 |  |
| erconsults | 2018-10-01 | 2024-06-30 | 117 |  |
| er | 2018-10-01 | 2024-06-30 | 117 |  |
| lab | 2018-10-01 | 2024-06-30 | 117 |  |
| derived_variables | 2018-10-01 | 2024-06-30 | 117 |  |
| vitals | 2018-10-01 | 2024-06-30 | 117 |  |
| radiology | 2018-10-01 | 2024-06-30 | 117 |  |
| roomtransfer | 2018-10-01 | 2024-06-30 | 117 |  |
| transfusion | 2018-10-01 | 2024-06-30 | 117 |  |
| ipscu | 2018-10-01 | 2024-06-30 | 117 |  |
| physicians | 2018-10-01 | 2024-06-30 | 117 |  |
| pharmacy | 2018-10-01 | 2024-06-30 | 117 |  |
| ipcmg | 2022-07-01 | 2024-06-30 | 117 |  |
| ipdiagnosis | 2018-10-01 | 2024-06-30 | 117 |  |
| erintervention | 2018-10-01 | 2024-06-30 | 117 |  |
| admdad | 2018-10-01 | 2024-06-30 | 117 |  |
| echo | 2022-07-02 | 2024-06-30 | 118 |  |
| locality_variables | 2018-10-01 | 2024-06-30 | 118 |  |
| ipintervention | 2018-10-01 | 2024-06-30 | 118 |  |
| iphig | 2018-10-01 | 2024-06-30 | 118 |  |
| erdiagnosis | 2018-10-01 | 2024-06-30 | 118 |  |
| erconsults | 2018-10-01 | 2024-06-30 | 118 |  |
| er | 2018-10-01 | 2024-06-30 | 118 |  |
| lab | 2018-10-01 | 2024-06-30 | 118 |  |
| derived_variables | 2018-10-01 | 2024-06-30 | 118 |  |
| vitals | 2018-10-01 | 2024-06-30 | 118 |  |
| radiology | 2018-10-01 | 2024-06-30 | 118 |  |
| roomtransfer | 2018-10-01 | 2024-06-30 | 118 |  |
| transfusion | 2018-10-01 | 2024-06-30 | 118 |  |
| ipscu | 2018-10-01 | 2024-06-30 | 118 |  |
| physicians | 2018-10-01 | 2024-06-30 | 118 |  |
| pharmacy | 2018-10-01 | 2024-06-30 | 118 |  |
| ipcmg | 2022-07-01 | 2024-06-30 | 118 |  |
| ipdiagnosis | 2018-10-01 | 2024-06-30 | 118 |  |
| erintervention | 2018-10-01 | 2024-06-30 | 118 |  |
| admdad | 2018-10-01 | 2024-06-30 | 118 |  |
| echo | None | None | 114 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 114 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 114 |  |
| iphig | 2015-04-01 | 2024-06-30 | 114 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 114 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 114 |  |
| er | 2015-04-01 | 2024-06-30 | 114 |  |
| lab | 2017-07-07 | 2024-06-30 | 114 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 114 |  |
| vitals | 2017-07-07 | 2024-06-30 | 114 |  |
| radiology | 2017-07-07 | 2024-06-30 | 114 |  |
| roomtransfer | None | None | 114 |  |
| transfusion | 2017-07-07 | 2024-06-30 | 114 |  |
| ipscu | 2015-04-01 | 2024-06-29 | 114 |  |
| physicians | 2015-04-01 | 2024-06-30 | 114 |  |
| pharmacy | 2017-07-08 | 2024-06-30 | 114 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 114 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 114 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 114 |  |
| admdad | 2015-04-01 | 2024-06-30 | 114 |  |
| echo | None | None | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| locality_variables | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| ipintervention | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| iphig | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| erdiagnosis | 2021-06-07 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| erconsults | 2021-06-07 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| er | 2021-06-07 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| lab | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| derived_variables | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| vitals | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| radiology | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| roomtransfer | None | None | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| transfusion | 2021-02-28 | 2024-06-29 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| ipscu | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| physicians | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| pharmacy | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| ipcmg | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| ipdiagnosis | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| erintervention | 2021-06-07 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| admdad | 2021-02-28 | 2024-06-30 | 115 | Hospital opened in February of 2021, and the ER opened in June of 2021. |
| echo | 2015-04-01 | 2024-06-30 | 116 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 116 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 116 |  |
| iphig | 2015-04-01 | 2024-06-30 | 116 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 116 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 116 |  |
| er | 2015-04-01 | 2024-06-30 | 116 |  |
| lab | 2015-04-01 | 2024-06-30 | 116 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 116 |  |
| vitals | 2015-04-01 | 2024-06-30 | 116 |  |
| radiology | 2015-04-01 | 2024-06-30 | 116 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 116 |  |
| transfusion | 2015-04-01 | 2024-06-30 | 116 |  |
| ipscu | 2015-04-01 | 2024-06-29 | 116 |  |
| physicians | 2015-04-01 | 2024-06-30 | 116 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 116 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 116 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 116 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 116 |  |
| admdad | 2015-04-01 | 2024-06-30 | 116 |  |
| echo | 2015-04-01 | 2015-04-27 | 131 | Hospital does not have a special care unit. |
| echo | 2015-05-26 | 2015-08-25 | 131 | Hospital does not have a special care unit. |
| echo | 2015-10-07 | 2016-04-06 | 131 | Hospital does not have a special care unit. |
| echo | 2016-05-17 | 2017-01-17 | 131 | Hospital does not have a special care unit. |
| echo | 2017-03-03 | 2018-06-19 | 131 | Hospital does not have a special care unit. |
| echo | 2018-08-01 | 2020-05-04 | 131 | Hospital does not have a special care unit. |
| echo | 2020-06-04 | 2023-02-03 | 131 | Hospital does not have a special care unit. |
| echo | 2023-05-30 | 2024-06-29 | 131 | Hospital does not have a special care unit. |
| locality_variables | 2015-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| ipintervention | 2015-04-03 | 2024-06-26 | 131 | Hospital does not have a special care unit. |
| iphig | 2016-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| erconsults | 2015-04-17 | 2015-04-17 | 131 | Hospital does not have a special care unit. |
| erconsults | 2015-06-02 | 2015-07-08 | 131 | Hospital does not have a special care unit. |
| erconsults | 2015-08-10 | 2015-10-06 | 131 | Hospital does not have a special care unit. |
| erconsults | 2015-11-08 | 2015-11-08 | 131 | Hospital does not have a special care unit. |
| erconsults | 2015-12-18 | 2015-12-18 | 131 | Hospital does not have a special care unit. |
| erconsults | 2016-04-14 | 2016-08-03 | 131 | Hospital does not have a special care unit. |
| erconsults | 2017-07-20 | 2020-11-11 | 131 | Hospital does not have a special care unit. |
| erconsults | 2020-12-11 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| er | 2015-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| lab | 2015-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| derived_variables | 2015-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| vitals | 2015-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| radiology | 2015-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| roomtransfer | 2015-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| transfusion | 2015-04-02 | 2015-06-15 | 131 | Hospital does not have a special care unit. |
| transfusion | 2015-08-17 | 2015-10-16 | 131 | Hospital does not have a special care unit. |
| transfusion | 2015-11-17 | 2015-11-22 | 131 | Hospital does not have a special care unit. |
| transfusion | 2015-12-22 | 2016-08-09 | 131 | Hospital does not have a special care unit. |
| transfusion | 2016-09-08 | 2016-10-21 | 131 | Hospital does not have a special care unit. |
| transfusion | 2016-12-07 | 2017-02-01 | 131 | Hospital does not have a special care unit. |
| transfusion | 2017-04-04 | 2017-04-24 | 131 | Hospital does not have a special care unit. |
| transfusion | 2017-05-25 | 2017-06-01 | 131 | Hospital does not have a special care unit. |
| transfusion | 2017-08-03 | 2017-08-03 | 131 | Hospital does not have a special care unit. |
| transfusion | 2017-09-19 | 2017-12-26 | 131 | Hospital does not have a special care unit. |
| transfusion | 2018-02-04 | 2018-06-08 | 131 | Hospital does not have a special care unit. |
| transfusion | 2018-07-17 | 2019-11-13 | 131 | Hospital does not have a special care unit. |
| transfusion | 2019-12-12 | 2020-02-12 | 131 | Hospital does not have a special care unit. |
| transfusion | 2020-04-04 | 2020-04-04 | 131 | Hospital does not have a special care unit. |
| transfusion | 2020-06-26 | 2020-08-08 | 131 | Hospital does not have a special care unit. |
| transfusion | 2020-09-21 | 2021-09-11 | 131 | Hospital does not have a special care unit. |
| transfusion | 2021-10-22 | 2022-03-09 | 131 | Hospital does not have a special care unit. |
| transfusion | 2022-04-07 | 2023-06-02 | 131 | Hospital does not have a special care unit. |
| transfusion | 2023-07-03 | 2024-01-25 | 131 | Hospital does not have a special care unit. |
| transfusion | 2024-03-12 | 2024-06-24 | 131 | Hospital does not have a special care unit. |
| ipscu | None | None | 131 | Hospital does not have a special care unit. |
| physicians | 2015-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| pharmacy | 2015-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| ipcmg | 2016-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| erintervention | 2015-04-02 | 2024-06-29 | 131 | Hospital does not have a special care unit. |
| admdad | 2015-04-01 | 2024-06-30 | 131 | Hospital does not have a special care unit. |
| echo | None | None | 108 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 108 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 108 |  |
| iphig | 2015-04-01 | 2024-06-30 | 108 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 108 |  |
| erconsults | 2019-11-01 | 2024-06-30 | 108 |  |
| er | 2015-04-01 | 2024-06-30 | 108 |  |
| lab | 2015-04-01 | 2024-06-30 | 108 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 108 |  |
| vitals | 2015-04-01 | 2024-06-30 | 108 |  |
| radiology | 2015-04-01 | 2024-06-30 | 108 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 108 |  |
| transfusion | 2015-04-01 | 2024-06-30 | 108 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 108 |  |
| physicians | 2015-04-01 | 2024-06-30 | 108 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 108 | Additional medication details found in column 'med_id_hospital_code_raw' |
| ipcmg | 2015-04-01 | 2024-06-30 | 108 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 108 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 108 |  |
| admdad | 2015-04-01 | 2024-06-30 | 108 |  |
| echo | None | None | 111 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 111 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 111 |  |
| iphig | 2015-04-01 | 2024-06-30 | 111 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 111 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 111 |  |
| er | 2015-04-01 | 2024-06-30 | 111 |  |
| lab | 2015-04-01 | 2024-06-30 | 111 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 111 |  |
| vitals | None | None | 111 |  |
| radiology | 2015-04-01 | 2024-06-30 | 111 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 111 |  |
| transfusion | 2015-04-02 | 2024-06-29 | 111 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 111 |  |
| physicians | 2015-04-01 | 2024-06-30 | 111 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 111 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 111 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 111 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 111 |  |
| admdad | 2015-04-01 | 2024-06-30 | 111 |  |
| echo | None | None | 112 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 112 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 112 |  |
| iphig | 2015-04-01 | 2024-06-30 | 112 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 112 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 112 |  |
| er | 2015-04-01 | 2024-06-30 | 112 |  |
| lab | 2015-04-01 | 2024-06-30 | 112 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 112 |  |
| vitals | None | None | 112 |  |
| radiology | 2015-04-01 | 2024-06-30 | 112 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 112 |  |
| transfusion | 2015-04-01 | 2024-06-30 | 112 |  |
| ipscu | 2015-04-01 | 2024-06-28 | 112 |  |
| physicians | 2015-04-01 | 2024-06-30 | 112 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 112 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 112 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 112 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 112 |  |
| admdad | 2015-04-01 | 2024-06-30 | 112 |  |
| echo | None | None | 113 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 113 |  |
| ipintervention | 2015-04-01 | 2024-06-29 | 113 |  |
| iphig | 2015-04-01 | 2024-06-30 | 113 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 113 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 113 |  |
| er | 2015-04-01 | 2024-06-30 | 113 |  |
| lab | 2015-04-01 | 2024-06-30 | 113 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 113 |  |
| vitals | None | None | 113 |  |
| radiology | 2015-04-01 | 2024-06-30 | 113 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 113 |  |
| transfusion | 2015-04-01 | 2024-06-29 | 113 |  |
| ipscu | 2015-04-01 | 2024-06-29 | 113 |  |
| physicians | 2015-04-01 | 2024-06-30 | 113 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 113 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 113 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 113 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 113 |  |
| admdad | 2015-04-01 | 2024-06-30 | 113 |  |
| echo | None | None | 122 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 122 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 122 |  |
| iphig | 2015-04-01 | 2024-06-30 | 122 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 122 |  |
| erconsults | 2015-04-01 | 2023-06-30 | 122 |  |
| er | 2015-04-01 | 2024-06-30 | 122 |  |
| lab | 2016-04-01 | 2024-06-30 | 122 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 122 |  |
| vitals | None | None | 122 |  |
| radiology | 2016-04-01 | 2024-06-30 | 122 |  |
| roomtransfer | None | None | 122 |  |
| transfusion | 2020-10-08 | 2024-06-29 | 122 |  |
| ipscu | 2015-04-01 | 2024-06-29 | 122 |  |
| physicians | 2015-04-01 | 2024-06-30 | 122 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 122 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 122 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 122 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 122 |  |
| admdad | 2015-04-01 | 2024-06-30 | 122 |  |
| echo | 2019-11-01 | 2021-06-30 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| echo | 2021-08-09 | 2022-05-30 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| locality_variables | 2019-11-01 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipintervention | 2019-11-01 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| iphig | 2019-11-01 | 2020-06-30 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| iphig | 2021-01-01 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erdiagnosis | 2019-12-03 | 2019-12-03 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erdiagnosis | 2020-01-29 | 2020-03-13 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erdiagnosis | 2020-05-29 | 2020-07-17 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erdiagnosis | 2020-08-15 | 2020-08-15 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erdiagnosis | 2020-10-30 | 2021-06-10 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erdiagnosis | 2021-08-20 | 2021-11-11 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erdiagnosis | 2021-12-22 | 2021-12-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erdiagnosis | 2022-05-25 | 2022-05-25 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erconsults | 2019-12-03 | 2019-12-03 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erconsults | 2020-01-29 | 2020-03-13 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erconsults | 2020-05-29 | 2020-07-17 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erconsults | 2020-08-15 | 2020-08-15 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erconsults | 2020-10-30 | 2021-06-10 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erconsults | 2021-08-20 | 2021-11-11 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erconsults | 2021-12-22 | 2021-12-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erconsults | 2022-05-25 | 2022-05-25 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| er | 2019-12-03 | 2019-12-03 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| er | 2020-01-29 | 2020-03-13 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| er | 2020-05-29 | 2020-07-17 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| er | 2020-08-15 | 2020-08-15 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| er | 2020-10-30 | 2021-06-10 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| er | 2021-08-20 | 2021-11-11 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| er | 2021-12-22 | 2021-12-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| er | 2022-05-25 | 2022-05-25 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| lab | 2019-11-01 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| derived_variables | 2019-11-01 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| vitals | 2019-11-01 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| radiology | 2019-11-01 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| roomtransfer | 2019-11-01 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| transfusion | 2019-11-01 | 2021-06-30 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| transfusion | 2021-08-03 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipscu | 2019-12-22 | 2019-12-22 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipscu | 2020-01-22 | 2020-02-06 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipscu | 2020-05-15 | 2020-05-15 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipscu | 2020-08-14 | 2020-08-14 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipscu | 2020-12-23 | 2020-12-24 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipscu | 2021-01-26 | 2021-01-26 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipscu | 2021-03-22 | 2021-04-02 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipscu | 2021-05-15 | 2021-08-09 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipscu | 2022-03-07 | 2022-03-23 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipscu | 2022-05-03 | 2022-05-03 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| physicians | 2019-11-01 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| pharmacy | 2019-11-01 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipcmg | None | None | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| ipdiagnosis | 2019-11-01 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erintervention | 2019-12-03 | 2019-12-03 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erintervention | 2020-01-31 | 2020-03-13 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erintervention | 2020-05-29 | 2020-07-17 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erintervention | 2020-08-15 | 2020-08-15 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erintervention | 2020-10-30 | 2021-06-10 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erintervention | 2021-08-20 | 2021-11-11 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erintervention | 2021-12-22 | 2021-12-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| erintervention | 2022-05-25 | 2022-05-25 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| admdad | 2019-11-01 | 2022-05-31 | 102 | Hospital provides specialized care without an ER of its own; ER through hospital 100. Low SCU encounters. |
| echo | 2019-10-31 | 2024-06-30 | 123 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 123 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 123 |  |
| iphig | 2015-04-01 | 2024-06-30 | 123 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 123 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 123 |  |
| er | 2015-04-01 | 2024-06-30 | 123 |  |
| lab | 2019-10-29 | 2024-06-30 | 123 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 123 |  |
| vitals | 2019-10-29 | 2024-06-30 | 123 |  |
| radiology | 2019-10-29 | 2024-06-30 | 123 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 123 |  |
| transfusion | 2019-10-29 | 2024-06-30 | 123 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 123 |  |
| physicians | 2015-04-01 | 2024-06-30 | 123 |  |
| pharmacy | 2019-10-29 | 2024-06-30 | 123 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 123 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 123 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 123 |  |
| admdad | 2015-04-01 | 2024-06-30 | 123 |  |
| echo | 2015-04-01 | 2024-06-30 | 107 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 107 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 107 |  |
| iphig | 2015-04-01 | 2024-06-30 | 107 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 107 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 107 |  |
| er | 2015-04-01 | 2024-06-30 | 107 |  |
| lab | 2015-04-01 | 2024-06-30 | 107 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 107 |  |
| vitals | None | None | 107 |  |
| radiology | 2015-04-01 | 2024-06-30 | 107 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 107 |  |
| transfusion | 2015-04-01 | 2024-06-30 | 107 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 107 |  |
| physicians | 2015-04-01 | 2024-06-30 | 107 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 107 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 107 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 107 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 107 |  |
| admdad | 2015-04-01 | 2024-06-30 | 107 |  |
| echo | 2021-12-03 | 2024-06-29 | 136 |  |
| locality_variables | 2021-12-03 | 2024-06-30 | 136 |  |
| ipintervention | 2021-12-03 | 2024-06-30 | 136 |  |
| iphig | 2021-12-03 | 2024-06-30 | 136 |  |
| erdiagnosis | 2021-12-06 | 2024-06-30 | 136 |  |
| erconsults | 2021-12-06 | 2024-06-30 | 136 |  |
| er | 2021-12-06 | 2024-06-30 | 136 |  |
| lab | 2021-12-03 | 2024-06-30 | 136 |  |
| derived_variables | 2021-12-03 | 2024-06-30 | 136 |  |
| vitals | 2021-12-03 | 2024-06-30 | 136 |  |
| radiology | 2021-12-03 | 2024-06-30 | 136 |  |
| roomtransfer | 2021-12-03 | 2024-06-30 | 136 |  |
| transfusion | 2021-12-07 | 2024-06-30 | 136 |  |
| ipscu | 2021-12-03 | 2024-06-30 | 136 |  |
| physicians | 2021-12-03 | 2024-06-30 | 136 |  |
| pharmacy | 2021-12-03 | 2024-06-30 | 136 |  |
| ipcmg | 2021-12-03 | 2024-06-30 | 136 |  |
| ipdiagnosis | 2021-12-03 | 2024-06-30 | 136 |  |
| erintervention | 2021-12-07 | 2024-06-30 | 136 |  |
| admdad | 2021-12-03 | 2024-06-30 | 136 |  |
| echo | 2021-12-03 | 2024-06-30 | 137 |  |
| locality_variables | 2021-12-02 | 2024-06-30 | 137 |  |
| ipintervention | 2021-12-02 | 2024-06-30 | 137 |  |
| iphig | 2021-12-02 | 2024-06-30 | 137 |  |
| erdiagnosis | 2021-12-05 | 2024-06-30 | 137 |  |
| erconsults | 2021-12-05 | 2024-06-30 | 137 |  |
| er | 2021-12-05 | 2024-06-30 | 137 |  |
| lab | 2021-12-03 | 2024-06-30 | 137 |  |
| derived_variables | 2021-12-02 | 2024-06-30 | 137 |  |
| vitals | 2021-12-02 | 2024-06-30 | 137 |  |
| radiology | 2021-12-04 | 2024-06-30 | 137 |  |
| roomtransfer | 2021-12-03 | 2024-06-30 | 137 |  |
| transfusion | 2021-12-09 | 2024-06-30 | 137 |  |
| ipscu | 2021-12-04 | 2024-06-30 | 137 |  |
| physicians | 2021-12-02 | 2024-06-30 | 137 |  |
| pharmacy | 2021-12-02 | 2024-06-30 | 137 |  |
| ipcmg | 2021-12-02 | 2024-06-30 | 137 |  |
| ipdiagnosis | 2021-12-02 | 2024-06-30 | 137 |  |
| erintervention | 2021-12-05 | 2024-06-30 | 137 |  |
| admdad | 2021-12-02 | 2024-06-30 | 137 |  |
| echo | 2021-12-03 | 2024-06-30 | 138 |  |
| locality_variables | 2021-12-02 | 2024-06-30 | 138 |  |
| ipintervention | 2021-12-03 | 2024-06-30 | 138 |  |
| iphig | 2021-12-02 | 2024-06-30 | 138 |  |
| erdiagnosis | 2021-12-05 | 2024-06-30 | 138 |  |
| erconsults | 2021-12-05 | 2024-06-30 | 138 |  |
| er | 2021-12-05 | 2024-06-30 | 138 |  |
| lab | 2021-12-03 | 2024-06-30 | 138 |  |
| derived_variables | 2021-12-02 | 2024-06-30 | 138 |  |
| vitals | 2021-12-02 | 2024-06-30 | 138 |  |
| radiology | 2021-12-04 | 2024-06-30 | 138 |  |
| roomtransfer | 2021-12-03 | 2024-06-30 | 138 |  |
| transfusion | 2021-12-08 | 2024-06-30 | 138 |  |
| ipscu | 2021-12-03 | 2024-06-30 | 138 |  |
| physicians | 2021-12-02 | 2024-06-30 | 138 |  |
| pharmacy | 2021-12-02 | 2024-06-30 | 138 |  |
| ipcmg | 2021-12-02 | 2024-06-30 | 138 |  |
| ipdiagnosis | 2021-12-02 | 2024-06-30 | 138 |  |
| erintervention | 2021-12-05 | 2024-06-30 | 138 |  |
| admdad | 2021-12-02 | 2024-06-30 | 138 |  |
| echo | 2015-04-03 | 2024-06-30 | 104 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 104 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 104 |  |
| iphig | 2015-04-01 | 2024-06-30 | 104 |  |
| erdiagnosis | 2015-04-03 | 2024-06-30 | 104 |  |
| erconsults | 2015-04-03 | 2024-06-30 | 104 |  |
| er | 2015-04-03 | 2024-06-30 | 104 |  |
| lab | 2015-04-01 | 2024-06-30 | 104 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 104 |  |
| vitals | 2015-04-01 | 2024-06-30 | 104 |  |
| radiology | 2015-04-01 | 2024-06-30 | 104 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 104 |  |
| transfusion | 2015-04-01 | 2024-06-30 | 104 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 104 |  |
| physicians | 2015-04-01 | 2024-06-30 | 104 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 104 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 104 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 104 |  |
| erintervention | 2015-04-03 | 2024-06-30 | 104 |  |
| admdad | 2015-04-01 | 2024-06-30 | 104 |  |
| echo | 2022-07-01 | 2024-06-30 | 124 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 124 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 124 |  |
| iphig | 2015-04-01 | 2024-06-30 | 124 |  |
| erdiagnosis | 2015-04-02 | 2024-06-30 | 124 |  |
| erconsults | 2015-04-02 | 2024-06-30 | 124 |  |
| er | 2015-04-02 | 2024-06-30 | 124 |  |
| lab | 2020-04-01 | 2024-06-30 | 124 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 124 |  |
| vitals | 2023-07-01 | 2024-06-30 | 124 |  |
| radiology | 2020-04-01 | 2024-06-30 | 124 |  |
| roomtransfer | 2022-01-01 | 2024-06-30 | 124 |  |
| transfusion | 2020-04-01 | 2024-06-30 | 124 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 124 |  |
| physicians | 2015-04-01 | 2024-06-30 | 124 |  |
| pharmacy | 2020-04-01 | 2024-06-30 | 124 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 124 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 124 |  |
| erintervention | 2015-04-03 | 2024-06-30 | 124 |  |
| admdad | 2015-04-01 | 2024-06-30 | 124 |  |
| echo | 2015-04-01 | 2024-06-30 | 103 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 103 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 103 |  |
| iphig | 2015-04-01 | 2024-06-30 | 103 |  |
| erdiagnosis | 2015-04-02 | 2024-06-30 | 103 |  |
| erconsults | 2015-04-02 | 2024-06-30 | 103 |  |
| er | 2015-04-02 | 2024-06-30 | 103 |  |
| lab | 2015-04-01 | 2024-06-30 | 103 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 103 |  |
| vitals | 2015-04-01 | 2024-06-30 | 103 |  |
| radiology | 2015-04-01 | 2024-06-30 | 103 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 103 |  |
| transfusion | 2015-04-01 | 2024-06-30 | 103 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 103 |  |
| physicians | 2015-04-01 | 2024-06-30 | 103 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 103 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 103 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 103 |  |
| erintervention | 2015-04-02 | 2024-06-30 | 103 |  |
| admdad | 2015-04-01 | 2024-06-30 | 103 |  |
| echo | None | None | 125 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 125 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 125 |  |
| iphig | 2015-04-01 | 2024-06-30 | 125 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 125 |  |
| erconsults | 2015-04-01 | 2023-09-30 | 125 |  |
| erconsults | 2024-04-01 | 2024-06-30 | 125 |  |
| er | 2015-04-01 | 2024-06-30 | 125 |  |
| lab | 2015-04-01 | 2024-06-30 | 125 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 125 |  |
| vitals | 2015-04-01 | 2024-06-30 | 125 |  |
| radiology | 2021-01-01 | 2021-03-31 | 125 |  |
| radiology | 2022-04-01 | 2024-06-30 | 125 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 125 |  |
| transfusion | 2015-04-01 | 2024-06-30 | 125 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 125 |  |
| physicians | 2015-04-01 | 2024-06-30 | 125 |  |
| pharmacy | None | None | 125 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 125 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 125 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 125 |  |
| admdad | 2015-04-01 | 2024-06-30 | 125 |  |
| echo | 2015-04-01 | 2024-06-30 | 126 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 126 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 126 |  |
| iphig | 2015-04-01 | 2024-06-30 | 126 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 126 |  |
| erconsults | 2015-04-02 | 2024-06-30 | 126 |  |
| er | 2015-04-01 | 2024-06-30 | 126 |  |
| lab | 2015-04-01 | 2024-06-30 | 126 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 126 |  |
| vitals | 2015-04-01 | 2024-06-30 | 126 |  |
| radiology | 2015-04-01 | 2024-06-30 | 126 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 126 |  |
| transfusion | 2015-04-02 | 2024-06-30 | 126 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 126 |  |
| physicians | 2015-04-01 | 2024-06-30 | 126 |  |
| pharmacy | 2017-04-01 | 2024-06-30 | 126 |  |
| ipcmg | 2015-04-01 | 2024-06-30 | 126 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 126 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 126 |  |
| admdad | 2015-04-01 | 2024-06-30 | 126 |  |
| echo | 2015-04-01 | 2020-07-31 | 105 |  |
| echo | 2020-11-01 | 2024-06-30 | 105 |  |
| locality_variables | 2015-04-01 | 2020-07-31 | 105 |  |
| locality_variables | 2020-11-01 | 2024-06-30 | 105 |  |
| ipintervention | 2015-04-01 | 2020-07-31 | 105 |  |
| ipintervention | 2020-11-01 | 2024-06-30 | 105 |  |
| iphig | 2017-11-01 | 2020-07-31 | 105 |  |
| iphig | 2020-11-01 | 2024-06-30 | 105 |  |
| erdiagnosis | 2015-04-01 | 2020-07-31 | 105 |  |
| erdiagnosis | 2020-11-01 | 2024-06-30 | 105 |  |
| erconsults | 2015-04-01 | 2020-07-31 | 105 |  |
| erconsults | 2020-11-01 | 2024-06-30 | 105 |  |
| er | 2015-04-03 | 2020-07-31 | 105 |  |
| er | 2020-11-01 | 2024-06-30 | 105 |  |
| lab | 2015-04-01 | 2020-07-31 | 105 |  |
| lab | 2020-11-01 | 2024-06-30 | 105 |  |
| derived_variables | 2015-04-01 | 2020-07-31 | 105 |  |
| derived_variables | 2020-11-01 | 2024-06-30 | 105 |  |
| vitals | 2017-11-01 | 2019-07-18 | 105 |  |
| vitals | 2019-09-17 | 2019-09-17 | 105 |  |
| vitals | 2019-11-01 | 2020-07-31 | 105 |  |
| vitals | 2020-11-01 | 2023-06-30 | 105 |  |
| radiology | 2015-04-01 | 2020-07-31 | 105 |  |
| radiology | 2020-11-01 | 2024-06-30 | 105 |  |
| roomtransfer | 2015-04-01 | 2020-07-31 | 105 |  |
| roomtransfer | 2020-11-01 | 2024-06-30 | 105 |  |
| transfusion | 2015-04-02 | 2020-07-31 | 105 |  |
| ipscu | 2015-04-05 | 2020-07-31 | 105 |  |
| ipscu | 2020-11-01 | 2024-06-30 | 105 |  |
| physicians | 2015-04-01 | 2020-07-31 | 105 |  |
| physicians | 2020-11-01 | 2024-06-30 | 105 |  |
| pharmacy | 2015-04-01 | 2019-07-18 | 105 |  |
| pharmacy | 2019-09-17 | 2019-09-17 | 105 |  |
| pharmacy | 2019-11-01 | 2020-07-31 | 105 |  |
| pharmacy | 2020-11-01 | 2024-06-30 | 105 |  |
| ipcmg | 2015-04-01 | 2019-10-31 | 105 |  |
| ipcmg | 2020-11-01 | 2024-06-30 | 105 |  |
| ipdiagnosis | 2015-04-01 | 2020-07-31 | 105 |  |
| ipdiagnosis | 2020-11-01 | 2024-06-30 | 105 |  |
| erintervention | 2015-04-01 | 2020-07-31 | 105 |  |
| erintervention | 2020-11-01 | 2024-06-30 | 105 |  |
| admdad | 2015-04-01 | 2020-07-31 | 105 |  |
| admdad | 2020-11-01 | 2024-06-30 | 105 |  |
| echo | 2015-04-01 | 2020-07-31 | 106 |  |
| echo | 2020-11-01 | 2024-06-30 | 106 |  |
| locality_variables | 2015-04-01 | 2020-07-31 | 106 |  |
| locality_variables | 2020-11-01 | 2024-06-30 | 106 |  |
| ipintervention | 2015-04-01 | 2020-07-31 | 106 |  |
| ipintervention | 2020-11-01 | 2024-06-30 | 106 |  |
| iphig | 2017-11-01 | 2020-07-31 | 106 |  |
| iphig | 2020-11-01 | 2024-06-30 | 106 |  |
| erdiagnosis | 2015-04-01 | 2020-07-31 | 106 |  |
| erdiagnosis | 2020-11-01 | 2024-06-30 | 106 |  |
| erconsults | 2015-04-01 | 2020-07-31 | 106 |  |
| erconsults | 2020-11-01 | 2024-06-30 | 106 |  |
| er | 2015-04-02 | 2020-07-31 | 106 |  |
| er | 2020-11-01 | 2024-06-30 | 106 |  |
| lab | 2015-04-01 | 2020-07-31 | 106 |  |
| lab | 2020-11-01 | 2024-06-30 | 106 |  |
| derived_variables | 2015-04-01 | 2020-07-31 | 106 |  |
| derived_variables | 2020-11-01 | 2024-06-30 | 106 |  |
| vitals | 2017-11-01 | 2020-07-31 | 106 |  |
| vitals | 2020-11-01 | 2023-06-30 | 106 |  |
| radiology | 2015-04-01 | 2020-07-31 | 106 |  |
| radiology | 2020-11-01 | 2024-06-30 | 106 |  |
| roomtransfer | 2015-04-01 | 2020-07-31 | 106 |  |
| roomtransfer | 2020-11-01 | 2024-06-30 | 106 |  |
| transfusion | 2015-04-10 | 2015-04-16 | 106 |  |
| transfusion | 2015-05-30 | 2015-05-30 | 106 |  |
| transfusion | 2015-07-21 | 2015-07-21 | 106 |  |
| transfusion | 2015-08-24 | 2015-08-24 | 106 |  |
| transfusion | 2015-09-25 | 2016-03-06 | 106 |  |
| transfusion | 2016-04-19 | 2016-04-27 | 106 |  |
| transfusion | 2016-06-05 | 2016-08-04 | 106 |  |
| transfusion | 2016-09-20 | 2016-12-26 | 106 |  |
| transfusion | 2017-03-02 | 2017-04-16 | 106 |  |
| transfusion | 2017-05-29 | 2017-08-07 | 106 |  |
| transfusion | 2017-09-12 | 2017-10-30 | 106 |  |
| ipscu | 2015-04-01 | 2020-07-31 | 106 |  |
| ipscu | 2020-11-01 | 2024-06-30 | 106 |  |
| physicians | 2015-04-01 | 2020-07-31 | 106 |  |
| physicians | 2020-11-01 | 2024-06-30 | 106 |  |
| pharmacy | 2015-04-01 | 2020-07-31 | 106 |  |
| pharmacy | 2020-11-01 | 2024-06-30 | 106 |  |
| ipcmg | 2015-04-01 | 2020-07-31 | 106 |  |
| ipcmg | 2020-11-01 | 2024-06-30 | 106 |  |
| ipdiagnosis | 2015-04-01 | 2020-07-31 | 106 |  |
| ipdiagnosis | 2020-11-01 | 2024-06-30 | 106 |  |
| erintervention | 2015-04-01 | 2020-07-31 | 106 |  |
| erintervention | 2020-11-01 | 2024-06-30 | 106 |  |
| admdad | 2015-04-01 | 2020-07-31 | 106 |  |
| admdad | 2020-11-01 | 2024-06-30 | 106 |  |
| echo | 2015-04-02 | 2022-05-31 | 100 |  |
| locality_variables | 2015-04-01 | 2022-05-31 | 100 |  |
| ipintervention | 2015-04-02 | 2022-05-31 | 100 |  |
| iphig | 2017-11-01 | 2022-05-31 | 100 |  |
| erdiagnosis | 2015-04-02 | 2022-05-31 | 100 |  |
| erconsults | 2015-04-02 | 2022-05-31 | 100 |  |
| er | 2015-04-02 | 2022-05-31 | 100 |  |
| lab | 2015-04-01 | 2022-05-31 | 100 |  |
| derived_variables | 2015-04-01 | 2022-05-31 | 100 |  |
| vitals | 2016-01-25 | 2022-05-31 | 100 |  |
| radiology | 2015-04-02 | 2022-05-31 | 100 |  |
| roomtransfer | 2015-04-01 | 2022-05-31 | 100 |  |
| transfusion | 2015-04-02 | 2022-05-31 | 100 |  |
| ipscu | 2015-04-04 | 2022-05-31 | 100 |  |
| physicians | 2015-04-01 | 2022-05-31 | 100 |  |
| pharmacy | 2015-04-01 | 2022-05-31 | 100 |  |
| ipcmg | 2015-04-01 | 2017-10-31 | 100 |  |
| ipdiagnosis | 2015-04-01 | 2022-05-31 | 100 |  |
| erintervention | 2015-04-02 | 2022-05-31 | 100 |  |
| admdad | 2015-04-01 | 2022-05-31 | 100 |  |
| echo | 2015-04-02 | 2022-05-31 | 101 |  |
| locality_variables | 2015-04-02 | 2022-05-31 | 101 |  |
| ipintervention | 2015-04-02 | 2022-05-31 | 101 |  |
| iphig | 2017-11-01 | 2022-05-31 | 101 |  |
| erdiagnosis | 2015-04-02 | 2022-05-31 | 101 |  |
| erconsults | 2015-04-02 | 2022-05-31 | 101 |  |
| er | 2015-04-02 | 2022-05-31 | 101 |  |
| lab | 2015-04-02 | 2022-05-31 | 101 |  |
| derived_variables | 2015-04-02 | 2022-05-31 | 101 |  |
| vitals | 2016-05-21 | 2016-05-25 | 101 |  |
| vitals | 2016-07-16 | 2016-07-16 | 101 |  |
| vitals | 2017-03-20 | 2017-03-20 | 101 |  |
| vitals | 2017-06-15 | 2017-06-30 | 101 |  |
| vitals | 2017-09-20 | 2022-05-31 | 101 |  |
| radiology | 2015-04-02 | 2022-05-31 | 101 |  |
| roomtransfer | 2015-04-02 | 2022-05-31 | 101 |  |
| transfusion | 2015-04-02 | 2022-05-30 | 101 |  |
| ipscu | 2015-04-02 | 2022-05-30 | 101 |  |
| physicians | 2015-04-02 | 2022-05-31 | 101 |  |
| pharmacy | 2015-04-02 | 2022-05-31 | 101 |  |
| ipcmg | 2015-04-02 | 2017-10-31 | 101 |  |
| ipdiagnosis | 2015-04-02 | 2022-05-31 | 101 |  |
| erintervention | 2015-04-02 | 2022-05-31 | 101 |  |
| admdad | 2015-04-02 | 2022-05-31 | 101 |  |
| echo | 2019-08-22 | 2024-06-30 | 119 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 119 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 119 |  |
| iphig | 2015-04-01 | 2024-06-30 | 119 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 119 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 119 |  |
| er | 2015-04-01 | 2024-06-30 | 119 |  |
| lab | 2015-04-01 | 2024-06-30 | 119 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 119 |  |
| vitals | 2015-04-01 | 2024-06-30 | 119 |  |
| radiology | 2015-04-01 | 2024-06-30 | 119 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 119 |  |
| transfusion | 2015-04-01 | 2024-06-30 | 119 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 119 |  |
| physicians | 2015-04-01 | 2024-06-30 | 119 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 119 |  |
| ipcmg | 2015-04-01 | 2022-06-30 | 119 |  |
| ipcmg | 2023-07-01 | 2024-06-30 | 119 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 119 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 119 |  |
| admdad | 2015-04-01 | 2024-06-30 | 119 |  |
| echo | 2019-08-23 | 2024-06-30 | 120 |  |
| locality_variables | 2015-04-01 | 2024-06-30 | 120 |  |
| ipintervention | 2015-04-01 | 2024-06-30 | 120 |  |
| iphig | 2015-04-01 | 2024-06-30 | 120 |  |
| erdiagnosis | 2015-04-01 | 2024-06-30 | 120 |  |
| erconsults | 2015-04-01 | 2024-06-30 | 120 |  |
| er | 2015-04-01 | 2024-06-30 | 120 |  |
| lab | 2015-04-01 | 2024-06-30 | 120 |  |
| derived_variables | 2015-04-01 | 2024-06-30 | 120 |  |
| vitals | 2015-04-01 | 2024-06-30 | 120 |  |
| radiology | 2015-04-01 | 2024-06-30 | 120 |  |
| roomtransfer | 2015-04-01 | 2024-06-30 | 120 |  |
| transfusion | 2015-04-01 | 2024-06-30 | 120 |  |
| ipscu | 2015-04-01 | 2024-06-30 | 120 |  |
| physicians | 2015-04-01 | 2024-06-30 | 120 |  |
| pharmacy | 2015-04-01 | 2024-06-30 | 120 |  |
| ipcmg | 2015-04-01 | 2022-06-30 | 120 |  |
| ipcmg | 2023-07-01 | 2024-06-30 | 120 |  |
| ipdiagnosis | 2015-04-01 | 2024-06-30 | 120 |  |
| erintervention | 2015-04-01 | 2024-06-30 | 120 |  |
| admdad | 2015-04-01 | 2024-06-30 | 120 |  |
| echo | 2022-04-26 | 2022-04-26 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| echo | 2022-07-06 | 2022-07-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| echo | 2022-08-18 | 2022-08-18 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| echo | 2023-04-21 | 2023-04-21 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| locality_variables | 2019-05-15 | 2021-05-07 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| locality_variables | 2021-09-23 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| locality_variables | 2022-03-26 | 2024-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2019-05-24 | 2019-05-24 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2019-07-07 | 2019-08-09 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2019-09-19 | 2019-09-19 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2019-12-09 | 2019-12-14 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2020-01-24 | 2020-01-24 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2020-03-10 | 2020-03-10 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2020-07-16 | 2020-07-28 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2020-09-23 | 2020-09-24 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2020-10-28 | 2020-10-28 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2020-12-10 | 2020-12-18 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2021-10-07 | 2021-10-31 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2021-12-09 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2022-05-12 | 2022-05-12 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2022-06-14 | 2022-07-04 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2022-08-03 | 2023-07-21 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2023-08-24 | 2023-10-24 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2023-12-13 | 2023-12-23 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipintervention | 2024-05-24 | 2024-05-24 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| iphig | 2019-05-15 | 2021-05-07 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| iphig | 2021-09-23 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| iphig | 2022-03-26 | 2024-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| erdiagnosis | None | None | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| erconsults | None | None | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| er | None | None | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| lab | 2019-05-15 | 2020-12-31 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| lab | 2021-10-05 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| lab | 2022-03-26 | 2024-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| derived_variables | 2019-05-15 | 2021-05-07 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| derived_variables | 2021-09-23 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| derived_variables | 2022-03-26 | 2024-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| vitals | 2019-05-15 | 2021-05-07 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| vitals | 2021-09-23 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| vitals | 2022-03-26 | 2024-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| radiology | 2019-07-09 | 2019-07-09 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| radiology | 2019-08-09 | 2019-08-21 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| radiology | 2019-09-19 | 2019-10-20 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| radiology | 2019-11-28 | 2020-12-22 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| radiology | 2021-10-07 | 2021-10-27 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| radiology | 2021-12-09 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| radiology | 2022-04-26 | 2023-07-21 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| radiology | 2023-08-19 | 2023-10-24 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| radiology | 2023-12-01 | 2024-01-03 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| radiology | 2024-02-09 | 2024-04-30 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| radiology | 2024-06-03 | 2024-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| roomtransfer | 2019-05-15 | 2020-12-31 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| roomtransfer | 2021-10-05 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| roomtransfer | 2022-03-26 | 2024-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| transfusion | 2020-07-07 | 2020-07-07 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| transfusion | 2023-08-10 | 2023-08-10 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipscu | None | None | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| physicians | 2019-05-15 | 2021-05-07 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| physicians | 2021-09-23 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| physicians | 2022-03-26 | 2024-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| pharmacy | 2019-05-15 | 2021-05-07 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| pharmacy | 2021-09-23 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| pharmacy | 2022-03-26 | 2024-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipcmg | 2019-05-15 | 2021-05-07 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipcmg | 2021-09-23 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipcmg | 2022-03-26 | 2022-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipcmg | 2023-07-05 | 2024-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipdiagnosis | 2019-05-15 | 2020-12-31 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipdiagnosis | 2021-10-05 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| ipdiagnosis | 2022-03-26 | 2024-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| erintervention | None | None | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| admdad | 2019-05-15 | 2021-05-07 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| admdad | 2021-09-23 | 2022-01-06 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |
| admdad | 2022-03-26 | 2024-06-29 | 121 | Hospital opened in May of 2019, provides specialized care and does not have an ER or a special care unit. |

### Encounters per year (`admdad_subset`)

| year | count |
| --- | --- |
| 2010 | <6 |
| 2012 | <6 |
| 2013 | 0 |
| 2014 | 0 |
| 2015 | 156000 |
| 2016 | 208000 |
| 2017 | 214000 |
| 2018 | 227000 |
| 2019 | 251000 |
| 2020 | 235000 |
| 2021 | 267000 |
| 2022 | 276000 |
| 2023 | 290000 |
| 2024 | 143000 |

### Lookup tables confirmed genuinely empty (real EXISTS check)

| table | genuinely empty |
| --- | --- |
| lookup_hospital | False |
| lookup_pharmacy_route | False |
| lookup_transfusion_concept | False |
| lookup_vitals_concept | False |
