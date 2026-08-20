# GEMINI schema report: `subdural_hematoma_v1_0_0`

## matview: `admdad_subset` (rows: 2268000)

| column | type |
| --- | --- |
| genc_id | integer |
| country | text |
| admitting_service_raw | text |
| discharging_service_raw | text |
| total_direct_cost | double precision |
| total_indirect_cost | double precision |
| total_cost | double precision |
| admit_category | text |
| discharge_disposition | integer |
| responsibility_for_payment | text |
| province_territory_issuing_health_card_number | text |
| number_of_alc_days | integer |
| institution_from | text |
| institution_to | text |
| readmission | text |
| residence_code | integer |
| gender | text |
| age | integer |
| mrp_service | text |
| entry_code | text |
| admission_date_time | text |
| discharge_date_time | text |
| admitting_service_mapped | text |
| discharging_service_mapped | text |
| patient_id_hashed | text |
| mrp_service_raw | text |
| patient_service_subservice | text |
| patient_service_subservice_mapped | text |
| mrp_service_mapped | text |
| admit_via_ambulance | text |
| alc_service_transfer_flag | text |
| blood_transfusion_indicator | text |
| main_patient_service_raw | text |
| main_patient_service_mapped | text |
| row_num | bigint |
| hospital_num | integer |

## matview: `cohort` (rows: 2268000)

| column | type |
| --- | --- |
| genc_id | integer |

## matview: `derived_variables_subset` (rows: 2268000)

| column | type |
| --- | --- |
| genc_id | integer |
| mlaps | double precision |
| admit_charlson_derived | integer |
| all_charlson_derived | integer |
| los_days_derived | double precision |
| readmission_30d_derived | boolean |
| readmission_7d_derived | boolean |
| readmission_30d_derived_cihi | boolean |
| readmission_7d_derived_cihi | boolean |
| in_hospital_mortality_derived | boolean |
| icu_entry_derived | boolean |
| icu_entry_in_24hr_derived | boolean |
| icu_entry_in_48hr_derived | boolean |
| icu_entry_in_72hr_derived | boolean |
| icu_los_days_derived | double precision |
| icu_los_hrs_derived | double precision |
| n_img_xray_derived | integer |
| n_img_ct_derived | integer |
| n_img_mri_derived | integer |
| n_img_us_derived | integer |
| n_img_int_derived | integer |
| n_rbc_transfusion_derived | integer |
| n_app_rbc_transfusion_derived | integer |
| n_routine_bloodwork_derived | integer |
| from_acute_care_institution_derived | boolean |
| to_acute_care_institution_derived | boolean |
| covid_icd_confirmed_derived | boolean |
| covid_icd_suspected_derived | boolean |
| epicare | integer |
| mlaps_24hrs | double precision |
| gim | boolean |
| all_med | boolean |
| hospital_num | integer |

## matview: `er_subset` (rows: 1859000)

| column | type |
| --- | --- |
| genc_id | integer |
| admit_via_ambulance | text |
| triage_level | text |
| ambulance_arrival_date_time | text |
| physician_initial_assessment_date_time | text |
| triage_date_time | text |
| disposition_date_time | text |
| left_er_date_time | text |
| registration_date_time | text |
| duration_er_stay_derived | double precision |
| cacs | text |
| ed_discharge_diagnosis | text |
| institution_number | text |
| institution_from | text |
| institution_to | text |
| visit_disposition | text |
| cacs_methodology_year | text |
| cacs_riw | text |
| cacs_riw_on | text |
| mac_code | text |
| blood_transfusion_indicator | text |
| referral_source_prior_to_ambulatory_care_visit | text |
| row_num | bigint |
| non_physician_initial_assessment_date_time | text |
| hospital_num | integer |

## matview: `erconsults_subset` (rows: 1899000)

| column | type |
| --- | --- |
| genc_id | integer |
| consult_occurrence | text |
| consult_service_code | text |
| consult_service_description | text |
| consult_request_date_time | text |
| consult_arrival_date_time | text |
| row_num | bigint |
| hospital_num | integer |

## matview: `erdiagnosis_subset` (rows: 5221000)

| column | type |
| --- | --- |
| genc_id | integer |
| er_diagnosis_code | text |
| er_diagnosis_type | text |
| row_num | bigint |
| hospital_num | integer |

## matview: `erintervention_subset` (rows: 2936000)

| column | type |
| --- | --- |
| genc_id | integer |
| intervention_type | integer |
| intervention_code | text |
| intervention_location_attribute | text |
| intervention_status_attribute | text |
| intervention_extent_attribute | text |
| out_of_hospital_indicator | text |
| intervention_episode_start_date_time | text |
| row_num | bigint |
| hospital_num | integer |

## matview: `ipcmg_subset` (rows: 2092000)

| column | type |
| --- | --- |
| genc_id | integer |
| methodology_year | integer |
| cmg | text |
| diagnosis_for_cmg_assignment | text |
| cmg_intervention | text |
| comorbidity_level | text |
| riw_inpatient_atypical_indicator | text |
| riw | double precision |
| row_num | bigint |
| hospital_num | integer |

## matview: `ipdiagnosis_subset` (rows: 13895000)

| column | type |
| --- | --- |
| genc_id | integer |
| diagnosis_code | text |
| diagnosis_cluster | text |
| diagnosis_type | text |
| diagnosis_prefix | text |
| row_num | bigint |
| hospital_num | integer |

## matview: `iphig_subset` (rows: 2242000)

| column | type |
| --- | --- |
| genc_id | integer |
| hig_methodology_year | smallint |
| hig_code | text |
| hig_description | text |
| hig_atypical_code | smallint |
| hig_atypical_code_desc | text |
| hig_weight | text |
| hig_elos | text |
| age_category | text |
| homecare_flag | smallint |
| scu_flag | smallint |
| cardioversion_flag | smallint |
| cell_saver_flag | smallint |
| chemotherapy_flag | smallint |
| dialysis_flag | smallint |
| feeding_tube_flag | smallint |
| heart_resuscitation_flag | smallint |
| invasive_ventilation_ge_96h_flag | smallint |
| invasive_ventilation_lt_96h_flag | smallint |
| paracentesis_flag | smallint |
| parenteral_nutrition_flag | smallint |
| pleurocentesis_flag | smallint |
| radiotherapy_flag | smallint |
| tracheostomy_flag | smallint |
| vascular_access_device_flag | smallint |
| row_num | bigint |
| hospital_num | integer |

## matview: `ipintervention_subset` (rows: 4329000)

| column | type |
| --- | --- |
| genc_id | integer |
| intervention_type | integer |
| intervention_code | text |
| procedure_location | text |
| intervention_location_attribute | text |
| intervention_status_attribute | text |
| intervention_extent_attribute | text |
| intervention_episode_start_date_time | text |
| row_num | bigint |
| hospital_num | integer |

## matview: `ipscu_subset` (rows: 632000)

| column | type |
| --- | --- |
| genc_id | integer |
| scu_admit_date_time | text |
| scu_discharge_date_time | text |
| icu_flag | boolean |
| row_num | bigint |
| scu_unit_number | integer |
| hospital_num | integer |

## matview: `lab_subset` (rows: 659381000)

| column | type |
| --- | --- |
| test_name_raw | text |
| test_code_raw | text |
| genc_id | integer |
| result_value | text |
| result_unit | text |
| collection_date_time | text |
| reference_range | text |
| test_type_mapped_omop | integer |
| row_num | bigint |
| hospital_num | integer |

## matview: `locality_variables_subset` (rows: 2268000)

| column | type |
| --- | --- |
| genc_id | integer |
| da16uid | integer |
| da11uid | integer |
| version | text |
| row_num | bigint |
| da21uid | integer |
| hospital_num | integer |

## matview: `lookup_cci` (rows: 17000)

| column | type |
| --- | --- |
| intervention_code | text |
| cci_short_title | text |
| cci_long_title | text |

## matview: `lookup_ccsr` (rows: 1000)

| column | type |
| --- | --- |
| ccsr | text |
| ccsr_desc | text |

## matview: `lookup_cihi_codes` (rows: 1000)

| column | type |
| --- | --- |
| table_name | text |
| column_name | text |
| value | text |
| description | text |
| version | text |

## matview: `lookup_data_coverage` (rows: 1000)

| column | type |
| --- | --- |
| data | text |
| min_date | date |
| max_date | date |
| hospital_num | integer |
| additional_info | text |

## matview: `lookup_hospital` (rows: 0)

| column | type |
| --- | --- |
| institution_id | smallint |
| hospital_num | integer |
| hospital_type | character varying(50) |
| additional_info | text |
| gim_cohort_avail | character varying(50) |
| other_med_cohort_avail | character varying(50) |
| icu_cohort_avail | character varying(50) |
| other_inpatient_cohort_avail | character varying(50) |

## matview: `lookup_icd10_ca_description` (rows: 19000)

| column | type |
| --- | --- |
| diagnosis_code | text |
| short_description | text |
| long_description | text |
| type | text |
| version | text |

## matview: `lookup_icd10_ca_to_ccsr` (rows: 82000)

| column | type |
| --- | --- |
| diagnosis_code | text |
| ccsr_default | text |
| ccsr_1 | text |
| ccsr_2 | text |
| ccsr_3 | text |
| ccsr_4 | text |
| ccsr_5 | text |
| ccsr_6 | text |
| gemini_derived | boolean |
| ccsr_version | text |

## matview: `lookup_lab_concept` (rows: 1000)

| column | type |
| --- | --- |
| concept_id | text |
| vocabulary_id | text |
| concept_desc | text |

## matview: `lookup_pharmacy_mapping` (rows: 15000)

| column | type |
| --- | --- |
| search_type | text |
| raw_input | text |
| rxnorm_match | text |
| drug_group | text |
| project_name | text |
| last_updated | timestamp with time zone |

## matview: `lookup_pharmacy_route` (rows: 0)

| column | type |
| --- | --- |
| route | text |
| route_administration | text |
| route_delivery | text |

## matview: `lookup_statcan_v2016` (rows: 48000)

| column | type |
| --- | --- |
| da16uid | integer |
| c16_popdw_popdens_sqkm | double precision |
| c16_inc_limat | double precision |
| c16_immcit | integer |
| c16_immcit_cancit | integer |
| c16_immcit_cancit_bel18 | integer |
| c16_immcit_cancit_18up | integer |
| c16_immcit_notcan | integer |
| c16_immsta | integer |
| c16_immsta_notimm | integer |
| c16_immsta_imm | integer |
| c16_immsta_imm5yrs | integer |
| c16_immsta_nonpr | integer |
| c16_ab | integer |
| c16_ab_singab | integer |
| c16_ab_singab_fn | integer |
| c16_ab_singab_met | integer |
| c16_ab_singab_in | integer |
| c16_ab_multab | integer |
| c16_ab_abnotelse | integer |
| c16_ab_nonab | integer |
| c16_vismin | integer |
| c16_vismin_not | integer |
| c16_eth_ab_nam | integer |
| c16_eth_other_nam | integer |
| c16_eth_eur | integer |
| c16_eth_car | integer |
| c16_eth_lat | integer |
| c16_eth_afr | integer |
| c16_eth_asi | integer |
| c16_eth_oce | integer |
| c16_ed_15over | integer |
| c16_ed_15over_nocert | integer |
| c16_ed_15over_secschool | integer |
| c16_ed_15over_postsec | integer |
| c16_ed_25to64 | integer |
| c16_ed_25to64_nocert | integer |
| c16_ed_25to64_secschool | integer |
| c16_ed_25to64_postsec | integer |
| c16_lab_ind_allind | integer |
| c16_lab_ind | integer |
| c16_lab_ind_notappl | integer |
| c16_lab_occ | integer |
| c16_lab_occ_notappl | integer |
| c16_lab_occ_allocc | integer |
| c16_lab_prh | integer |
| c16_lab_prh_wfh | integer |
| c16_lab_prh_out_of_can | integer |
| c16_lab_prh_nfaddr_work | integer |
| c16_lab_prh_usplace | integer |
| c16_lab_class | integer |
| c16_lab_class_allcl | integer |
| c16_lab_class_allcl_empl | integer |
| c16_lab_class_allcl_selfem | integer |
| c16_lab_avg_weeks_worked | double precision |
| c16_lab_wact_work | integer |
| c16_lab_wact_full | integer |
| c16_lab_wact_part | integer |
| c16_lab_wact_dnw | integer |
| c16_lab_wact | integer |
| c16_lab_labf | integer |
| c16_lab_labf_inlf | integer |
| c16_lab_labf_unempl | integer |
| c16_lab_labf_empl | integer |
| c16_lab_labf_notin | integer |
| c16_lab_part_rate | double precision |
| c16_lab_empl_rate | double precision |
| c16_lab_unempl_rate | double precision |
| instability_da16 | double precision |
| instability_q_da16 | integer |
| deprivation_da16 | double precision |
| deprivation_q_da16 | integer |
| dependency_da16 | double precision |
| dependency_q_da16 | integer |
| ethniccon_da16 | double precision |
| ethniccon_q_da16 | integer |
| cm_atype | text |
| btippe | integer |
| atippe | integer |
| qabtippe | integer |
| qnbtippe | integer |
| dabtippe | integer |
| dnbtippe | integer |
| qaatippe | integer |
| qnatippe | integer |
| daatippe | integer |
| dnatippe | integer |
| impflg | text |
| c16_inc_total_u5_ct | integer |
| c16_inc_total_5to10_ct | integer |
| c16_inc_total_10to15_ct | integer |
| c16_inc_total_15to20_ct | integer |
| c16_inc_total_20to25_ct | integer |
| c16_inc_total_25to30_ct | integer |
| c16_inc_total_30to35_ct | integer |
| c16_inc_total_30to40_ct | integer |
| c16_inc_total_40to45_ct | integer |
| c16_inc_total_45to50_ct | integer |
| c16_inc_total_50to60_ct | integer |
| c16_inc_total_60to70_ct | integer |
| c16_inc_total_70to80_ct | integer |
| c16_inc_total_80to90_ct | integer |
| c16_inc_total_90to100_ct | integer |
| c16_inc_total_100to125_ct | integer |
| c16_inc_total_125to150_ct | integer |
| c16_inc_total_150to200_ct | integer |
| c16_inc_total_200up_ct | integer |
| c16_inc_num_ct | integer |
| ice_inc_c16_ct | double precision |

## matview: `lookup_statcan_v2021` (rows: 58000)

| column | type |
| --- | --- |
| da21uid | integer |
| c21_prov_code | integer |
| c21_prov_name | text |
| c21_cd_code | integer |
| c21_cd_name | text |
| c21_da_name | integer |
| c21_popdw_pop21 | integer |
| c21_popdw_pop16 | integer |
| c21_popdw_pop_perc_change | double precision |
| c21_popdw_privdw_total | integer |
| c21_popdw_privdw_usres | integer |
| c21_popdw_popdens_sqkm | double precision |
| c21_land_area_sqkm | double precision |
| c21_inc_stats | integer |
| c21_inc_total_grps | integer |
| c21_inc_num | integer |
| c21_inc_med | double precision |
| c21_inc_total | integer |
| c21_inc_total_u5 | integer |
| c21_inc_total_5to10 | integer |
| c21_inc_total_10to15 | integer |
| c21_inc_total_100up | integer |
| c21_inc_total_90to100 | integer |
| c21_inc_total_70to80 | integer |
| c21_inc_total_60to70 | integer |
| c21_inc_total_50to60 | integer |
| c21_inc_total_45to50 | integer |
| c21_inc_total_15to20 | integer |
| c21_inc_total_20to25 | integer |
| c21_inc_total_100to125 | integer |
| c21_inc_total_125to150 | integer |
| c21_inc_total_150to200 | integer |
| c21_inc_total_200up | integer |
| c21_inc_med_total_fam | double precision |
| c21_inc_aft_tax_med | double precision |
| c21_prev_inc_limat | double precision |
| c21_inc_licoat | integer |
| c21_prev_inc_licoat | double precision |
| c21_incavg_num | integer |
| c21_incavg_avg | double precision |
| c21_immcit | integer |
| c21_immcit_cancit | integer |
| c21_immcit_cancit_bel18 | integer |
| c21_immcit_cancit_18up | integer |
| c21_immcit_notcan | integer |
| c21_immsta | integer |
| c21_immsta_notimm | integer |
| c21_immsta_imm | integer |
| c21_immsta_imm_11to21 | integer |
| c21_immsta_imm_11to15 | integer |
| c21_immsta_imm_16to21 | integer |
| c21_immsta_nonpr | integer |
| c21_ind | integer |
| c21_ind_id | integer |
| c21_ind_id_singind | integer |
| c21_ind_id_singind_fn | integer |
| c21_ind_id_singind_met | integer |
| c21_ind_id_singind_inuit | integer |
| c21_ind_id_multind | integer |
| c21_ind_id_indnotelse | integer |
| c21_ind_id_nonind | integer |
| c21_vismin | integer |
| c21_vismin_not | integer |
| c21_eth_ab_nam | integer |
| c21_eth_eur | integer |
| c21_eth_asi | integer |
| c21_eth_afr | integer |
| c21_eth_lat | integer |
| c21_eth_car | integer |
| c21_eth_other_nam | integer |
| c21_eth_cult_orig | integer |
| c21_ed_15over | integer |
| c21_ed_15over_nocert | integer |
| c21_ed_15over_secschool | integer |
| c21_ed_15over_postsec | integer |
| c21_ed_25to64 | integer |
| c21_ed_25to64_nocert | integer |
| c21_ed_25to64_secschool | integer |
| c21_ed_25to64_postsec | integer |
| c21_lab_occ | integer |
| c21_lab_occ_notappl | integer |
| c21_lab_occ_allocc | integer |
| c21_lab_ind | integer |
| c21_lab_ind_allind | integer |
| c21_lab_ind_notappl | integer |
| c21_lab_class | integer |
| c21_lab_class_allcl | integer |
| c21_lab_class_allcl_empl | integer |
| c21_lab_class_allcl_selfem | integer |
| c21_lab_wact | integer |
| c21_lab_wact_work | integer |
| c21_lab_wact_dnw | integer |
| c21_lab_wact_full | integer |
| c21_lab_wact_part | integer |
| c21_lab_avg_weeks_worked | double precision |
| c21_lab_labf | integer |
| c21_lab_labf_inlf | integer |
| c21_lab_labf_notin | integer |
| c21_lab_labf_empl | integer |
| c21_lab_labf_unempl | integer |
| c21_lab_part_rate | double precision |
| c21_lab_empl_rate | double precision |
| c21_lab_unempl_rate | double precision |
| c21_lab_plw | integer |
| c21_lab_plw_wfh | integer |
| c21_lab_plw_out_of_can | integer |
| c21_lab_plw_nfaddr_work | integer |
| c21_lab_plw_usplace | integer |
| c21_inc_total_25to30 | integer |
| c21_inc_total_30to35 | integer |
| c21_inc_total_40to45 | integer |
| c21_inc_total_35to40 | integer |
| Pop2021 | double precision |
| households_dwellings_DA21 | double precision |
| material_resources_DA21 | double precision |
| age_labourforce_DA21 | double precision |
| racialized_NC_pop_DA21 | double precision |
| households_dwellings_q_DA21 | double precision |
| material_resources_q_DA21 | double precision |
| age_labourforce_q_DA21 | double precision |
| racialized_NC_pop_q_DA21 | double precision |
| btippe | integer |
| atippe | integer |
| qabtippe | integer |
| qnbtippe | integer |
| dabtippe | integer |
| dnbtippe | integer |
| qaatippe | integer |
| qnatippe | integer |
| daatippe | integer |
| dnatippe | integer |
| impflg | integer |
| popctrraclass | text |

## matview: `lookup_transfer_subset` (rows: 1250000)

| column | type |
| --- | --- |
| genc_id | integer |
| institution_from_mns | text |
| institution_to_mns | text |
| institution_to_type_mns | text |
| institution_from_type_mns | text |

## matview: `lookup_transfusion_concept` (rows: 0)

| column | type |
| --- | --- |
| concept_id | integer |
| vocabulary_id | text |
| concept_desc | text |

## matview: `lookup_vitals_concept` (rows: 0)

| column | type |
| --- | --- |
| concept_id | text |
| vocabulary_id | text |
| concept_desc | text |

## matview: `pharmacy_subset` (rows: 84266000)

| column | type |
| --- | --- |
| genc_id | integer |
| med_id_generic_name_raw | text |
| med_id_brand_name_raw | text |
| med_id_din | text |
| med_id_ndc | text |
| med_id_ahfs | text |
| med_id_ahfs_description_raw | text |
| med_id_hospital_code_raw | text |
| dose_amount | text |
| dose_unit | text |
| route | text |
| frequency | text |
| order_description | text |
| order_number | text |
| finalvolume | text |
| rate | text |
| iv_component_type | text |
| PRN_IND | text |
| strength_dose | text |
| med_start_date_time | text |
| med_end_date_time | text |
| medication_suspend_start_date_time | text |
| medication_suspend_end_date_time | text |
| row_num | bigint |
| hospital_num | integer |

## matview: `physicians_subset` (rows: 2267000)

| column | type |
| --- | --- |
| genc_id | integer |
| admitting_physician_gim | text |
| discharging_physician_gim | text |
| mrp_gim | text |
| mrp_cpso_hashed | character varying(64) |
| adm_phy_cpso_hashed | character varying(64) |
| dis_phy_cpso_hashed | character varying(64) |
| row_num | bigint |
| hospital_num | integer |

## matview: `radiology_subset` (rows: 8687000)

| column | type |
| --- | --- |
| genc_id | integer |
| test_name_raw | text |
| ordered_date_time | text |
| performed_date_time | text |
| modality_mapped | text |
| body_part_mapped | text |
| imaging_order_id | text |
| modality_raw | text |
| row_num | bigint |
| imaging_result | text |
| hospital_num | integer |

## matview: `rxnorm_cache` (rows: 713000)

| column | type |
| --- | --- |
| rxcui | integer |
| score | double precision |
| search_type | text |
| raw_input | text |
| manually_added | boolean |
| date_added | text |
| active | boolean |
| row_num | bigint |

## matview: `vitals_subset` (rows: 411962000)

| column | type |
| --- | --- |
| genc_id | integer |
| measurement_name | text |
| measure_date_time | text |
| measurement_value | text |
| reference_range | text |
| measurement_mapped_omop | integer |
| measurement_unit | text |
| row_num | bigint |
| hospital_num | integer |
