use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, DataStruct, DeriveInput};

pub fn derive_proc_macro_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let DeriveInput {
        ident: struct_name_ident,
        data,
        ..
    }: DeriveInput = parse_macro_input!(input as DeriveInput);

    // Only generate code for struct.
    if let syn::Data::Struct(data_struct) =  data {
        // get the class name Partial{struct name}
        let partial_settings_struct_name = proc_macro2::Ident::new(&(String::from("Partial") + &struct_name_ident.to_string()), struct_name_ident.span());

        let partial_settings_struct_content = transform_fields_into_partial_struct_fields(&data_struct);
        let combine_function = transform_fields_into_combine_function(&data_struct);
        let try_from_internals = transform_fields_into_try_from_internals(&data_struct);
        let from_internals = transform_fields_into_from_internals(&data_struct);
        let names_and_types = transform_fields_into_names_and_types(&data_struct);
        let to_strings_internals = transform_fields_to_strings_internals(&data_struct);
        let partial_to_strings_internals = transform_fields_partial_to_strings_internals(&data_struct);

      quote! {

        // Make PartialSettings struct
        // each field  gets an entry
        // field : Option<Type> 
        // if labeled optional or merge use ( must be an option if optional already)
        // field : Type
        #[cfg_attr(feature = "json_schema_gen", derive(JsonSchema))]
        #[derive(Serialize, Deserialize, Debug, Clone, Default)]
        /// Partial Version of the struct #struct_name_ident
        pub struct #partial_settings_struct_name {

            #partial_settings_struct_content

        }

        // create the combine function
        // each field  gets an entry
        // field: self.field.or(other.field),
        // if labeled merge ( merge is a trait that allows for custom merging)
        // field: self.field.merge(other.field),

        impl Combine for #partial_settings_struct_name {
            #combine_function
        }

        impl TryFrom<#partial_settings_struct_name> for  #struct_name_ident {
            type Error = PartialConvertError;

            fn try_from(value: #partial_settings_struct_name) -> Result<#struct_name_ident, PartialConvertError> {
                Ok(#struct_name_ident {
                    #try_from_internals
                })
            }
        }

        impl  #struct_name_ident{
            fn get_names_and_types() -> Vec<(String,String)> {
                #names_and_types
            }
        }

        impl SettingsPrint for #struct_name_ident{
            fn to_strings(&self) -> Vec<String>{
                // returns vec![ 
                // "setting = value"
                // "setting.subsetting = value "
                // ]

                let mut settings_vec = vec![];
                #to_strings_internals
                settings_vec
            }
        }

        impl SettingsPrint for #partial_settings_struct_name{
            fn to_strings(&self) -> Vec<String>{
                // returns vec![ 
                // "setting = value"
                // "setting.subsetting = value "
                // ]

                let mut settings_vec = vec![];
                #partial_to_strings_internals
                settings_vec
            }
        }


        impl From<#struct_name_ident> for  #partial_settings_struct_name {
            fn from(value: #struct_name_ident) -> Self {
                #partial_settings_struct_name {
                    #from_internals
                }
            }
        }


      }

    } else {
        quote! {}
    }
    .into()
}

// Make PartialSettings struct
// each field  gets an entry
// field : Option<Type>
// if labeled optional or merge use ( must be an option if optional already)
// field : Type

fn transform_fields_into_partial_struct_fields(data_struct: &DataStruct) -> TokenStream {
    match data_struct.fields {
        syn::Fields::Named(ref fields) => {
            let props_ts_iter = fields.named.iter().map(|named_field| {
                let field_ident = named_field.ident.as_ref().unwrap();
                let type_ident_original = &named_field.ty;

                let mut optional = false;
                let mut recursive_type_opt = None;

                for attribute in &named_field.attrs {
                    if attribute.path().is_ident("Optional") {
                        optional = true;
                    }
                    if attribute.path().is_ident("Recursive") {
                        recursive_type_opt = Some(attribute.parse_args::<syn::Type>().unwrap());
                    }
                }

                if optional {
                    quote! {
                        #field_ident : #type_ident_original,
                    }
                } else if let Some(recursive_type) = recursive_type_opt {
                    quote! {
                        #field_ident : Option<#recursive_type>,
                    }
                } else {
                    quote! {
                        #field_ident : Option<#type_ident_original>,
                    }
                }
            });
            // Unwrap iterator into a [proc_macro2::TokenStream].
            quote! {
                #(#props_ts_iter)*
            }
        }
        _ => quote! {},
    }
}

// create the combine function
// each field  gets an entry
// field: self.field.or(other.field),
// if labeled combine ( combine is a trait that allows for custom combining)
// field: self.field.merge(other.field),
fn transform_fields_into_combine_function(data_struct: &DataStruct) -> TokenStream {
    match data_struct.fields {
        syn::Fields::Named(ref fields) => {
            let props_ts_iter = fields.named.iter().map(|named_field| {
                let field_ident = named_field.ident.as_ref().unwrap();
                quote! {
                    self.#field_ident.combine(other.#field_ident);
                }
            });
            // Unwrap iterator into a [proc_macro2::TokenStream].
            quote! {
                fn combine(&mut self, mut other: Self) {
                    #(#props_ts_iter)*
                }

            }
        }
        _ => quote! {},
    }
}

fn transform_fields_into_try_from_internals(data_struct: &DataStruct) -> TokenStream {
    match data_struct.fields {
        syn::Fields::Named(ref fields) => {
            let props_ts_iter = fields
                .named
                .iter()
                .map(|named_field| {
                    let field_ident = named_field.ident.as_ref().unwrap();
                    let type_ident_original = &named_field.ty;


                    let field_ident_str = named_field.ident.as_ref().unwrap().to_string();

                    let mut optional = false;
                    let mut allow = false;
                    let mut recursive_type_opt = None;

                    for attribute in &named_field.attrs {
                        if attribute.path().is_ident("Optional") {
                            optional = true;
                        }
                        if attribute.path().is_ident("Recursive") {
                            recursive_type_opt = Some(attribute.parse_args::<syn::Type>().unwrap());
                        }
                        if attribute.path().is_ident("AllowDefault") {
                            allow =true;
                        }
                    }

                    if optional {
                        quote! {
                            #field_ident: value.#field_ident,
                        }
                    } else if recursive_type_opt.is_some() {

                        if allow {
                            quote! {

                                #field_ident: #type_ident_original::try_from(value.#field_ident.unwrap_or_default())
                                    .map_err(|err| PartialConvertError(  #field_ident_str.to_string() + &err.0 ))?,
                            }
                        } else {
                            quote! {
                                #field_ident: #type_ident_original::try_from(
                                    value
                                        .#field_ident
                                        .ok_or(
                                            PartialConvertError(  #field_ident_str.to_string())
                                        )?
                                    )
                                    .map_err(|err| PartialConvertError(  #field_ident_str.to_string() + &err.0 ))?,
                            }
                        }
                    }
                    else if allow {
                        quote! {
                            #field_ident: value.#field_ident.unwrap_or_default(),
                        }
                    }
                    else{
                        quote! {
                            #field_ident: value.#field_ident.ok_or(PartialConvertError(#field_ident_str.to_string()))?,

                        }
                    }
                });
            // Unwrap iterator into a [proc_macro2::TokenStream].
            quote! {

                #(#props_ts_iter)*

            }
        }
        _ => quote! {},
    }
}

fn transform_fields_into_from_internals(data_struct: &DataStruct) -> TokenStream {
    match data_struct.fields {
        syn::Fields::Named(ref fields) => {
            let props_ts_iter = fields.named.iter().map(|named_field| {
                let field_ident = named_field.ident.as_ref().unwrap();

                let mut optional = false;
                let mut recursive_type_opt = None;

                for attribute in &named_field.attrs {
                    if attribute.path().is_ident("Optional") {
                        optional = true;
                    }

                    if attribute.path().is_ident("Recursive") {
                        recursive_type_opt = Some(attribute.parse_args::<syn::Type>().unwrap());
                    }
                }

                if optional {
                    quote! {
                        #field_ident: value.#field_ident,
                    }
                } else if let Some(recursive_type) = recursive_type_opt {
                    quote! {
                        #field_ident : Some(#recursive_type::from(value.#field_ident)),
                    }
                } else {
                    quote! {
                        #field_ident: Some(value.#field_ident),
                    }
                }
            });
            // Unwrap iterator into a [proc_macro2::TokenStream].
            quote! {

                 #(#props_ts_iter)*

            }
        }
        _ => quote! {},
    }
}

fn transform_fields_into_names_and_types(data_struct: &DataStruct) -> TokenStream {
    match data_struct.fields {
        syn::Fields::Named(ref fields) => {
            let props_ts_iter = fields.named.iter().map(|named_field| {
                let field_ident = named_field.ident.as_ref()
                    .expect("This is needed at build time").to_string();
                let type_ident_original = &named_field.ty.to_token_stream().to_string();
                quote! {
                   ( #field_ident.to_string(), #type_ident_original.to_string()),
                }
            });
            // Unwrap iterator into a [proc_macro2::TokenStream].
            quote! {

                vec![
                    #(#props_ts_iter)*
                ]


            }
        }
        _ => quote! {},
    }
}

fn transform_fields_to_strings_internals(data_struct: &DataStruct) -> proc_macro2::TokenStream {
    match data_struct.fields {
        syn::Fields::Named(ref fields) => {
            let props_ts_iter = fields.named.iter().map(|named_field| {
                let field_ident = named_field.ident.as_ref().unwrap();

                let field_ident_str = named_field.ident.as_ref().unwrap().to_string();

                let mut optional = false;
                let mut custom_print = false;
                let mut recursive_type_opt = None;

                for attribute in &named_field.attrs {
                    if attribute.path().is_ident("Optional") {
                        optional = true;
                    }
                    if attribute.path().is_ident("Recursive") {
                        recursive_type_opt = Some(attribute.parse_args::<syn::Type>().unwrap());
                    }

                    if attribute.path().is_ident("CustomPrint") {
                        custom_print = true;
                    }
                }

                if custom_print {
                    quote! {
                        settings_vec.extend(
                            self.#field_ident
                                .to_strings()
                                .iter()
                                .map(|x| format!("{} = {}", #field_ident_str, x))
                        );
                    }
                } else if optional {
                    if recursive_type_opt.is_some() {
                        quote! {
                            if let Some(set) = self.#field_ident.as_ref(){
                                for s in set.to_strings(){
                                    settings_vec.push(format!("{}.{}",#field_ident_str, s));
                                }
                            }
                        }
                    } else {
                        quote! {
                            if let Some(set) = self.#field_ident{
                                settings_vec.push(format!("{} = {}", #field_ident_str, set));
                            }
                        }
                    }
                } else if recursive_type_opt.is_some() {
                    quote! {
                        for s in self.#field_ident.to_strings(){
                            settings_vec.push(format!("{}.{}",#field_ident_str, s));
                        }
                    }
                } else {
                    quote! {
                        settings_vec.push(format!("{} = {}", #field_ident_str, self.#field_ident));
                    }
                }
            });
            // Unwrap iterator into a [proc_macro2::TokenStream].
            quote! {

                 #(#props_ts_iter)*

            }
        }
        _ => quote! {},
    }
}

fn transform_fields_partial_to_strings_internals(
    data_struct: &DataStruct,
) -> proc_macro2::TokenStream {
    match data_struct.fields {
        syn::Fields::Named(ref fields) => {
            let props_ts_iter = fields.named.iter().map(|named_field| {
                let field_ident = named_field.ident.as_ref().unwrap();

                let field_ident_str = named_field.ident.as_ref().unwrap().to_string();

                let mut custom_print = false;
                let mut recursive_type_opt = None;

                for attribute in &named_field.attrs {
                    if attribute.path().is_ident("Recursive") {
                        recursive_type_opt = Some(attribute.parse_args::<syn::Type>().unwrap());
                    }

                    if attribute.path().is_ident("CustomPrint") {
                        custom_print = true;
                    }
                }

                if custom_print {
                    quote! {
                        if let Some(set) = self.#field_ident.as_ref(){
                                settings_vec.extend(
                                    set
                                        .to_strings()
                                        .iter()
                                        .map(|x| format!("{} = {}", #field_ident_str, x))
                                );
                        }
                    }
                } else if recursive_type_opt.is_some() {
                    quote! {
                        if let Some(set) = self.#field_ident.as_ref(){
                            for s in set.to_strings(){
                                settings_vec.push(format!("{}.{}",#field_ident_str, s));
                            }
                        }
                    }
                } else {
                    quote! {
                        if let Some(set) = self.#field_ident.as_ref(){
                            settings_vec.push(format!("{} = {}", #field_ident_str, set));
                        }
                    }
                }
            });
            // Unwrap iterator into a [proc_macro2::TokenStream].
            quote! {

                 #(#props_ts_iter)*

            }
        }
        _ => quote! {},
    }
}
