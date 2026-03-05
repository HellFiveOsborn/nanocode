use arboard::Clipboard;
use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use image::{DynamicImage, ImageFormat, RgbaImage};
use std::io::Cursor;

pub enum ClipboardPayload {
    ImageDataUrl(String),
    Text(String),
}

pub fn read_clipboard_payload() -> Result<ClipboardPayload, String> {
    let mut clipboard = Clipboard::new().map_err(|err| format!("clipboard indisponivel: {err}"))?;

    if let Ok(image) = clipboard.get_image() {
        let width = u32::try_from(image.width)
            .map_err(|_| "largura da imagem no clipboard excede limite suportado".to_string())?;
        let height = u32::try_from(image.height)
            .map_err(|_| "altura da imagem no clipboard excede limite suportado".to_string())?;
        let raw = image.bytes.into_owned();
        let expected = width as usize * height as usize * 4;
        if raw.len() != expected {
            return Err("formato de imagem do clipboard invalido (esperado RGBA8)".to_string());
        }
        let rgba = RgbaImage::from_raw(width, height, raw)
            .ok_or_else(|| "nao foi possivel montar imagem RGBA do clipboard".to_string())?;
        let dyn_image = DynamicImage::ImageRgba8(rgba);
        let mut cursor = Cursor::new(Vec::new());
        dyn_image
            .write_to(&mut cursor, ImageFormat::Png)
            .map_err(|err| format!("falha ao codificar PNG do clipboard: {err}"))?;
        let png = cursor.into_inner();
        let encoded = STANDARD.encode(png);
        return Ok(ClipboardPayload::ImageDataUrl(format!(
            "data:image/png;base64,{encoded}"
        )));
    }

    if let Ok(text) = clipboard.get_text() {
        return Ok(ClipboardPayload::Text(text));
    }

    Err("clipboard nao contem texto ou imagem".to_string())
}
