use polars::prelude::*;
use spreadsheet_ods::read_ods;
use std::collections::HashMap;
use std::collections::VecDeque;
//---------------------------------------------------------------------------------------------------------
// STRUCTS
//---------------------------------------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AssumptionSet {
    pub mort: DataFrame,
    pub lapse: DataFrame,
    pub inf: DataFrame,
    pub acq: DataFrame,
    pub mtn: DataFrame,
    pub spot: DataFrame,
    pub load: DataFrame,
}

//---------------------------------------------------------------------------------------------------------
// PRIVATE
//---------------------------------------------------------------------------------------------------------

// Helper function to get a sheet by name from the ODS document
fn __get_sheet_by_name<'a>(
    doc: &'a spreadsheet_ods::WorkBook,
    sheet_name: &str,
) -> Option<&'a spreadsheet_ods::Sheet> {
    for idx in 0..doc.num_sheets() {
        let wsh = doc.sheet(idx);
        if wsh.name() == sheet_name {
            return Some(wsh);
        }
    }

    None
}

// Helper to parse a cell's text content (removes Text(…) and quotes)
fn ___parse_cell_text(cell_str: &str) -> String {
    if cell_str.contains("Text(") {
        let start = cell_str.find("Text(").unwrap_or(0) + 5;
        let end = if let Some(style_pos) = cell_str.find("), style:") {
            style_pos
        } else {
            cell_str.rfind(")").unwrap_or(cell_str.len())
        };
        let extracted = &cell_str[start..end];
        // Remove quotes if present
        if extracted.starts_with('"') && extracted.ends_with('"') {
            extracted[1..extracted.len() - 1].to_string()
        } else {
            extracted.to_string()
        }
    } else {
        cell_str.to_string()
    }
}

// The first column is always included in addtion to col
fn __get_indices_names_hashmap(
    sheet: &spreadsheet_ods::Sheet,
    col_names: &[&str],
    new_col_names: Option<&[&str]>,
) -> PolarsResult<HashMap<usize, (String, String)>> {
    let mut indices = HashMap::new();

    // Get the header row (assume it's the first row)
    let mut header: Vec<String> = Vec::new();
    let mut col_idx = 0;
    loop {
        if let Some(cell) = sheet.cell(0, col_idx) {
            let cell_str = format!("{:?}", cell);
            let name = ___parse_cell_text(&cell_str);
            header.push(name);
            col_idx += 1;
        } else {
            break;
        }
    }

    // Always include the first column (index 0)
    if let Some(first_col_name) = header.get(0) {
        indices.insert(0, (first_col_name.clone(), first_col_name.clone()));
    }

    // Find indices for requested columns
    for (i, &col_name) in col_names.iter().enumerate() {
        let mut found = false;
        for (idx, name) in header.iter().enumerate() {
            if name == col_name {
                let new_name = if let Some(new_names) = new_col_names {
                    new_names
                        .get(i)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| name.clone())
                } else {
                    name.clone()
                };
                indices.insert(idx, (name.clone(), new_name));
                found = true;
                break;
            }
        }
        if !found {
            return Err(PolarsError::ComputeError(
                format!("Column '{}' not found in sheet header", col_name).into(),
            ));
        }
    }

    Ok(indices)
}

fn __parse_col_by_index_to_f64(
    sheet: &spreadsheet_ods::Sheet,
    col_idx: usize,
) -> PolarsResult<Vec<f64>> {
    let mut col_data = Vec::new();

    let mut row_idx = 1; // Skip header

    loop {
        if let Some(cell_content) = sheet.cell(row_idx, col_idx as u32) {
            let cell_str = format!("{:?}", cell_content);
            let val = if cell_str.contains("Number(") {
                let start = cell_str.find("Number(").unwrap_or(0) + 7;
                let end = cell_str.rfind(")").unwrap_or(cell_str.len());
                cell_str[start..end].parse::<f64>().unwrap_or(0.0)
            } else {
                0.0
            };
            col_data.push(val);
            row_idx += 1;
        } else {
            break;
        }
    }
    Ok(col_data)
}

fn _get_assumption_df(
    sheet_name: &str,
    col_names: &[&str],
    new_col_names: Option<&[&str]>,
) -> PolarsResult<DataFrame> {
    // If new_col_names is provided, ensure it matches the length of col_names
    if let Some(ref new_names) = new_col_names {
        if col_names.len() != new_names.len() {
            return Err(PolarsError::ComputeError(
                "Length of col_names and new_col_names must match".into(),
            ));
        }
    }

    // Read the ODS file
    let doc = read_ods("src/assumptions/assumptions.ods")
        .map_err(|e| PolarsError::ComputeError(format!("Failed to read ODS file: {}", e).into()))?;

    // Find the sheet index by name
    let sheet = __get_sheet_by_name(&doc, sheet_name).ok_or_else(|| {
        PolarsError::ComputeError(format!("Sheet '{}' not found in ODS file", sheet_name).into())
    })?;

    // Get indices for requested columns (excluding first column)
    let col_hashmap = __get_indices_names_hashmap(sheet, col_names, new_col_names)?;

    let mut series_vec = VecDeque::new();

    for (col_idx, (_, new_name)) in col_hashmap.iter() {
        // Convert to f64
        let col_data_f64 = __parse_col_by_index_to_f64(sheet, *col_idx)?
            .into_iter()
            .collect::<Vec<f64>>();

        // First column is always i32
        if *col_idx == 0 {
            let col_data_i32: Vec<i32> = col_data_f64.into_iter().map(|x| x as i32).collect();
            let col = Series::new(new_name.into(), col_data_i32).into_column();
            series_vec.push_front(col)
        } else {
            let col = Series::new(new_name.into(), col_data_f64).into_column();
            series_vec.push_back(col);
        };
    }

    DataFrame::new(series_vec.into())
}

//---------------------------------------------------------------------------------------------------------
// PUBLIC
//---------------------------------------------------------------------------------------------------------
// Mortality assumption: The schema is slightly different from other since it is based on gender
pub fn get_mort_df(mort_type: &str) -> PolarsResult<DataFrame> {
    let col1 = format!("{}_m", mort_type);
    let col2 = format!("{}_f", mort_type);
    let col_names = [col1.as_str(), col2.as_str()];
    _get_assumption_df("mort_rate", &col_names, None)
}

// Lapse assumption
pub fn get_lapse_df(lapse_type: &str) -> PolarsResult<DataFrame> {
    _get_assumption_df("lapse_rate", &[lapse_type], Some(&["lapse_rate"]))
}

// Inflation assumption
pub fn get_inf_rate_df(inf_type: &str) -> PolarsResult<DataFrame> {
    _get_assumption_df("inf_rate", &[inf_type], Some(&["inf_rate"]))
}

// Acquisition assumption
pub fn get_acq_exp_df(acq_exp_type: &str) -> PolarsResult<DataFrame> {
    _get_assumption_df("acq_exp", &[acq_exp_type], Some(&["real_acq_exp_pp"]))
}

// Maintenance assumption
pub fn get_mtn_exp_df(mtn_exp_type: &str) -> PolarsResult<DataFrame> {
    _get_assumption_df("mtn_exp", &[mtn_exp_type], Some(&["real_mtn_exp_pp"]))
}

pub fn get_spot_rate_df(spot_rate_type: &str) -> PolarsResult<DataFrame> {
    _get_assumption_df("spot_rate", &[spot_rate_type], Some(&["spot_rate"]))
}

pub fn get_load_rate_df(load_rate_type: &str) -> PolarsResult<DataFrame> {
    _get_assumption_df("load_rate", &[load_rate_type], Some(&["load_rate"]))
}

//---------------------------------------------------------------------------------------------------------
// UNIT TESTS
//---------------------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_assumption_df_basic() {
        // Test reading data from the lapse_rate sheet
        let sheet_name = "lapse_rate";
        let col_names = ["lapse_01"];
        let result = _get_assumption_df(sheet_name, &col_names, None);

        match result {
            Ok(df) => println!("{:?}", df),
            Err(e) => println!("Error: {:?}", e),
        }
        assert!(true, "Assumption DataFrame should be created successfully");
    }
}
