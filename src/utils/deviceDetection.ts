import type { ViewMode } from '../types';

interface NavigatorUAData {
  mobile?: boolean;
}

export interface DeviceDetectionResult {
  isMobileOrTablet: boolean;
  recommendedViewMode: ViewMode;
}

/**
 * Automatically detects whether the user's device and viewport should render
 * the mobile Material 3 Android UI or the desktop 3-pane Studio UI.
 */
export function detectDeviceUI(): DeviceDetectionResult {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    return { isMobileOrTablet: false, recommendedViewMode: 'desktop-3pane' };
  }

  const ua = navigator.userAgent || '';
  const uaDataMobile =
    (navigator as Navigator & { userAgentData?: NavigatorUAData }).userAgentData?.mobile === true;
  const isTouchMac = navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1;
  const isMobileOrTabletUA =
    uaDataMobile ||
    isTouchMac ||
    /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile|Tablet|Silk|Kindle/i.test(
      ua
    );

  const isNarrowViewport = window.innerWidth < 768;
  const isTouchOnlyScreen =
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(pointer: coarse) and (hover: none)').matches &&
    navigator.maxTouchPoints > 0 &&
    window.innerWidth < 1280;

  const isMobileOrTablet = isMobileOrTabletUA || isNarrowViewport || isTouchOnlyScreen;

  return {
    isMobileOrTablet,
    recommendedViewMode: isMobileOrTablet ? 'android-ui' : 'desktop-3pane',
  };
}
