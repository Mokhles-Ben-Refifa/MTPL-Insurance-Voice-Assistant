import polars as pl
import logging

logger = logging.getLogger(__name__)


MIN_EXPECTED_ROWS = 16_000_000

vehicle_column_mapping = {
    "Kenteken": "vehicle_number_plate",
    "Voertuigsoort": "vehicle_usage_type",
    "Merk": "vehicle_maker",
    "Handelsbenaming": "vehicle_model",
    "Vervaldatum APK": "vehicle_inspection_expiry_DELETE",
    "Datum tenaamstelling": "vehicle_registration_DELETE",
    "Bruto BPM": "vehicle_gross_bpm_value",
    "Inrichting": "vehicle_body_type",
    "Aantal zitplaatsen": "vehicle_number_of_seats",
    "Eerste kleur": "vehicle_primary_color",
    "Tweede kleur": "vehicle_secondary_color",
    "Aantal cilinders": "vehicle_number_of_cylinders",
    "Cilinderinhoud": "vehicle_engine_size",
    "Massa ledig voertuig": "vehicle_net_weight",
    "Toegestane maximum massa voertuig": "vehicle_gross_weight",
    "Massa rijklaar": "vehicle_ready_to_drive_weight",
    "Maximum massa trekken ongeremd": "vehicle_max_towing_weight_unbraked",
    "Maximum trekken massa geremd": "vehicle_max_towing_weight_braked",
    "Datum eerste toelating": "vehicle_first_registration_DELETE",
    "Datum eerste tenaamstelling in Nederland": "vehicle_netherlands_first_registration_DELETE",
    "Wacht op keuren": "vehicle_awaiting_inspection",
    "Catalogusprijs": "vehicle_value_new",
    "WAM verzekerd": "vehicle_is_insured",
    "Maximale constructiesnelheid": "vehicle_max_construction_speed",
    "Laadvermogen": "vehicle_payload",
    "Oplegger geremd": "vehicle_has_braked_trailer",
    "Aanhangwagen autonoom geremd": "vehicle_has_autonomous_braked_trailer",
    "Aanhangwagen middenas geremd": "vehicle_has_center_axle_braked_trailer",
    "Aantal staanplaatsen": "vehicle_number_of_standing_spots",
    "Aantal deuren": "vehicle_number_of_doors",
    "Aantal wielen": "vehicle_number_of_wheels",
    "Afstand hart koppeling tot achterzijde voertuig": "vehicle_coupling_to_rear_distance",
    "Afstand voorzijde voertuig tot hart koppeling": "vehicle_coupling_to_front_distance",
    "Afwijkende maximum snelheid": "vehicle_max_deviating_speed",
    "Lengte": "vehicle_length",
    "Breedte": "vehicle_width",
    "Europese voertuigcategorie": "vehicle_eu_category",
    "Europese voertuigcategorie toevoeging": "vehicle_eu_category_addition",
    "Europese uitvoeringcategorie toevoeging": "vehicle_eu_version_addition",
    "Plaats chassisnummer": "vehicle_chassis_number_location",
    "Technische max. massa voertuig": "vehicle_technical_gross_weight",
    "Type": "vehicle_type",
    "Type gasinstallatie": "vehicle_gas_system_type",
    "Typegoedkeuringsnummer": "vehicle_type_approval_code",
    "Variant": "vehicle_engine_code",
    "Uitvoering": "vehicle_version_code",
    "Volgnummer wijziging EU typegoedkeuring": "vehicle_eu_type_approval_revision",
    "Vermogen massarijklaar": "vehicle_power_to_net_weight_ratio",
    "Wielbasis": "vehicle_wheelbase",
    "Export indicator": "vehicle_is_marked_for_export",
    "Openstaande terugroepactie indicator": "vehicle_has_open_recall",
    "Vervaldatum tachograaf": "vehicle_tachograph_expiry_DELETE",
    "Taxi indicator": "vehicle_is_taxi",
    "Maximum massa samenstelling": "vehicle_max_combined_weight",
    "Aantal rolstoelplaatsen": "vehicle_number_of_wheelchair_spaces",
    "Maximum ondersteunende snelheid": "vehicle_max_supported_speed",
    "Jaar laatste registratie tellerstand": "vehicle_year_of_last_odometer_report",
    "Tellerstandoordeel": "vehicle_odometer_verdict",
    "Code toelichting tellerstandoordeel": "vehicle_odometer_verdict_code",
    "Tenaamstellen mogelijk": "vehicle_can_be_registered",
    "Vervaldatum APK DT": "vehicle_inspection_expiry_date",
    "Datum tenaamstelling DT": "vehicle_last_registration_date",
    "Datum eerste toelating DT": "vehicle_first_registration_date",
    "Datum eerste tenaamstelling in Nederland DT": "vehicle_netherlands_first_registration_date",
    "Vervaldatum tachograaf DT": "vehicle_tachograph_expiry_date",
    "Maximum last onder de vooras(sen) (tezamen)/koppeling": "vehicle_max_load_front_axle_or_coupling_kg",
    "Type remsysteem voertuig code": "vehicle_brake_system_code",
    "Rupsonderstelconfiguratiecode": "vehicle_tracked_chassis_code",
    "Wielbasis voertuig minimum": "vehicle_min_wheelbase",
    "Wielbasis voertuig maximum": "vehicle_max_wheelbase",
    "Lengte voertuig minimum": "vehicle_min_length",
    "Lengte voertuig maximum": "vehicle_max_length",
    "Breedte voertuig minimum": "vehicle_min_width",
    "Breedte voertuig maximum": "vehicle_max_width",
    "Hoogte voertuig": "vehicle_height",
    "Hoogte voertuig minimum": "vehicle_min_height",
    "Hoogte voertuig maximum": "vehicle_max_height",
    "Massa bedrijfsklaar minimaal": "vehicle_min_operational_weight",
    "Massa bedrijfsklaar maximaal": "vehicle_max_operational_weight",
    "Technisch toelaatbaar massa koppelpunt": "vehicle_max_coupling_load",
    "Maximum massa technisch maximaal": "vehicle_max_technical_weight",
    "Maximum massa technisch minimaal": "vehicle_min_technical_weight",
    "Subcategorie Nederland": "vehicle_netherlands_subcategory",
    "Verticale belasting koppelpunt getrokken voertuig": "vehicle_vertical_coupling_load_towed",
    "Zuinigheidsclassificatie": "vehicle_fuel_efficiency_rating",
    "Registratie datum goedkeuring (afschrijvingsmoment BPM)": "vehicle_bpm_depreciation_DELETE",
    "Registratie datum goedkeuring (afschrijvingsmoment BPM) DT": "vehicle_bpm_depreciation_date",
    "Gemiddelde Lading Waarde": "vehicle_average_cargo_value",
    "Aerodynamische voorziening of uitrusting": "vehicle_has_aero_features",
    "Additionele massa alternatieve aandrijving": "vehicle_extra_weight_alternative_drive",
    "Verlengde cabine indicator": "vehicle_has_extended_cabin",
    "API Gekentekende_voertuigen_assen": "vehicle_api_axles_DELETE",
    "API Gekentekende_voertuigen_brandstof": "vehicle_api_fuel_DELETE",
    "API Gekentekende_voertuigen_carrosserie": "vehicle_api_body_DELETE",
    "API Gekentekende_voertuigen_carrosserie_specifiek": "vehicle_api_body_details_DELETE",
    "API Gekentekende_voertuigen_voertuigklasse": "vehicle_api_class_DELETE",
}

vehicle_dtypes_mapping = {
    "vehicle_number_plate": pl.Utf8,
    "vehicle_usage_type": pl.Utf8,
    "vehicle_maker": pl.Utf8,
    "vehicle_model": pl.Utf8,
    "vehicle_gross_bpm_value": pl.Float32,
    "vehicle_body_type": pl.Utf8,
    "vehicle_number_of_seats": pl.Float32,
    "vehicle_primary_color": pl.Utf8,
    "vehicle_secondary_color": pl.Utf8,
    "vehicle_number_of_cylinders": pl.Float32,
    "vehicle_engine_size": pl.Float32,
    "vehicle_net_weight": pl.Float32,
    "vehicle_gross_weight": pl.Float32,
    "vehicle_ready_to_drive_weight": pl.Float32,
    "vehicle_max_towing_weight_unbraked": pl.Float32,
    "vehicle_max_towing_weight_braked": pl.Float32,
    "vehicle_awaiting_inspection": pl.Utf8,
    "vehicle_value_new": pl.Float32,
    "vehicle_is_insured": pl.Utf8,
    "vehicle_max_construction_speed": pl.Float32,
    "vehicle_payload": pl.Utf8,
    "vehicle_has_braked_trailer": pl.Utf8,
    "vehicle_has_autonomous_braked_trailer": pl.Utf8,
    "vehicle_has_center_axle_braked_trailer": pl.Utf8,
    "vehicle_number_of_standing_spots": pl.Utf8,
    "vehicle_number_of_doors": pl.Float32,
    "vehicle_number_of_wheels": pl.Float32,
    "vehicle_coupling_to_rear_distance": pl.Utf8,
    "vehicle_coupling_to_front_distance": pl.Utf8,
    "vehicle_max_deviating_speed": pl.Utf8,
    "vehicle_length": pl.Float32,
    "vehicle_width": pl.Float32,
    "vehicle_eu_category": pl.Utf8,
    "vehicle_eu_category_addition": pl.Utf8,
    "vehicle_eu_version_addition": pl.Utf8,
    "vehicle_chassis_number_location": pl.Utf8,
    "vehicle_technical_gross_weight": pl.Float32,
    "vehicle_type": pl.Utf8,
    "vehicle_gas_system_type": pl.Utf8,
    "vehicle_type_approval_code": pl.Utf8,
    "vehicle_engine_code": pl.Utf8,
    "vehicle_version_code": pl.Utf8,
    "vehicle_eu_type_approval_revision": pl.Float32,
    "vehicle_power_to_net_weight_ratio": pl.Float32,
    "vehicle_wheelbase": pl.Float32,
    "vehicle_is_marked_for_export": pl.Utf8,
    "vehicle_has_open_recall": pl.Utf8,
    "vehicle_is_taxi": pl.Utf8,
    "vehicle_max_combined_weight": pl.Float32,
    "vehicle_number_of_wheelchair_spaces": pl.Utf8,
    "vehicle_max_supported_speed": pl.Utf8,
    "vehicle_year_of_last_odometer_report": pl.Float32,
    "vehicle_odometer_verdict": pl.Utf8,
    "vehicle_odometer_verdict_code": pl.Utf8,
    "vehicle_can_be_registered": pl.Utf8,
    "vehicle_inspection_expiry_date": pl.Utf8,
    "vehicle_last_registration_date": pl.Utf8,
    "vehicle_first_registration_date": pl.Utf8,
    "vehicle_netherlands_first_registration_date": pl.Utf8,
    "vehicle_tachograph_expiry_date": pl.Utf8,
    "vehicle_max_load_front_axle_or_coupling_kg": pl.Utf8,
    "vehicle_brake_system_code": pl.Utf8,
    "vehicle_tracked_chassis_code": pl.Utf8,
    "vehicle_min_wheelbase": pl.Utf8,
    "vehicle_max_wheelbase": pl.Utf8,
    "vehicle_min_length": pl.Utf8,
    "vehicle_max_length": pl.Utf8,
    "vehicle_min_width": pl.Utf8,
    "vehicle_max_width": pl.Utf8,
    "vehicle_height": pl.Float32,
    "vehicle_min_height": pl.Utf8,
    "vehicle_max_height": pl.Utf8,
    "vehicle_min_operational_weight": pl.Utf8,
    "vehicle_max_operational_weight": pl.Utf8,
    "vehicle_max_coupling_load": pl.Utf8,
    "vehicle_max_technical_weight": pl.Utf8,
    "vehicle_min_technical_weight": pl.Utf8,
    "vehicle_netherlands_subcategory": pl.Utf8,
    "vehicle_vertical_coupling_load_towed": pl.Utf8,
    "vehicle_fuel_efficiency_rating": pl.Utf8,
    "vehicle_bpm_depreciation_date": pl.Utf8,
    "vehicle_average_cargo_value": pl.Utf8,
    "vehicle_has_aero_features": pl.Utf8,
    "vehicle_extra_weight_alternative_drive": pl.Utf8,
    "vehicle_has_extended_cabin": pl.Utf8,
    "vehicle_model_original": pl.Utf8,
    "vehicle_maker_original": pl.Utf8,
}


def clean_number_plate_column(
    df: pl.DataFrame, column: str = "vehicle_number_plate"
) -> pl.DataFrame:
    result = df.with_columns(
        pl.col(column).str.replace_all("-", "").str.replace_all(" ", "").alias(column)
    )
    logger.debug(f"Number plate cleaned in column '{column}'")
    return result


def apply_base_model(df: pl.DataFrame, model_col: str) -> pl.DataFrame:
    result = df.with_columns(
        pl.col(model_col)
        .str.to_lowercase()
        .str.replace(r"[^\w\s\.]", " ", literal=False)
        .str.replace(r"\s+", " ", literal=False)
        .str.replace(r"^\s+|\s+$", "", literal=False)
        .str.extract(r"^([^ ]+)")
        .alias(model_col)
    )
    return result


def process_cars(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Starting car data processing")

    # Data validation
    logger.info("Running data validation check...")
    actual_rows = df.height

    if actual_rows < MIN_EXPECTED_ROWS:
        error_message = (
            f"Data validation failed! Cars dataset has only {actual_rows} rows, "
            f"which is less than the minimum threshold of {MIN_EXPECTED_ROWS}."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    logger.info(f"Data validation passed: Found {actual_rows} rows.")

    logger.info("Renaming CARS columns...")

    # Optional: Check unmapped columns (for safety)
    original_cols = set(df.columns)
    mapped_cols = set(vehicle_column_mapping.keys())
    unmapped_cols = original_cols - mapped_cols

    if unmapped_cols:
        logger.warning(
            f"These columns are not in the mapping and will remain unchanged: {unmapped_cols}"
        )

    # Rename using the mapping
    df = df.rename(vehicle_column_mapping)

    logger.info("Creating backup of original columns...")
    df = df.with_columns(
        pl.col("vehicle_model").alias("vehicle_model_original"),
        pl.col("vehicle_maker").alias("vehicle_maker_original"),
    )

    # Drop columns ending with _DELETE
    cols_before = len(df.columns)
    df = df.select([col for col in df.columns if not col.endswith("_DELETE")])
    cols_after = len(df.columns)
    logger.info(f"Dropped {cols_before - cols_after} columns with _DELETE suffix")

    df = df.select(
        [
            pl.col(col).cast(dtype, strict=False)
            if col in df.columns
            else pl.lit(None).cast(dtype, strict=False).alias(col)
            for col, dtype in vehicle_dtypes_mapping.items()
        ]
    )

    # Clean vehicle number plate column
    df = clean_number_plate_column(df)

    # Filter for 'Personenauto' usage type
    df = df.filter(pl.col("vehicle_usage_type") == "Personenauto")

    # Normalize vehicle_model column
    df = apply_base_model(df, "vehicle_model")

    # Log duplicates count
    num_duplicates = df.filter(pl.col("vehicle_number_plate").is_duplicated()).height
    logger.info(f"Number of duplicated vehicle_number_plate rows: {num_duplicates}")

    return df
