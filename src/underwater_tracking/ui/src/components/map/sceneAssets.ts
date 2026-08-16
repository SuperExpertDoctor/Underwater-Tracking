export const SCENE_ASSET_URLS = {
  background: "/assets/scene/background.png",
  carrier: "/assets/scene/carrier.png",
  uuv: "/assets/scene/uuv.png",
  submarine: "/assets/scene/submarine.png",
} as const;

export type SceneAssets = Record<keyof typeof SCENE_ASSET_URLS, HTMLImageElement | null>;
export type ImageLoader = (url: string) => Promise<HTMLImageElement | null>;

function loadImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => resolve(null);
    image.src = url;
  });
}

export async function loadSceneAssets(loader: ImageLoader = loadImage): Promise<SceneAssets> {
  const load = (url: string) => loader(url).catch(() => null);
  const [background, carrier, uuv, submarine] = await Promise.all([
    load(SCENE_ASSET_URLS.background),
    load(SCENE_ASSET_URLS.carrier),
    load(SCENE_ASSET_URLS.uuv),
    load(SCENE_ASSET_URLS.submarine),
  ]);

  return { background, carrier, uuv, submarine };
}

export function coverImageRect(imageWidth: number, imageHeight: number, width: number, height: number) {
  const scale = Math.max(width / imageWidth, height / imageHeight);
  const scaledWidth = imageWidth * scale;
  const scaledHeight = imageHeight * scale;

  return {
    x: (width - scaledWidth) / 2,
    y: (height - scaledHeight) / 2,
    width: scaledWidth,
    height: scaledHeight,
  };
}
