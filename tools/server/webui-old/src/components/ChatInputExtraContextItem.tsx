import {
  DocumentTextIcon,
  SpeakerWaveIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { MessageExtra } from '../utils/types';
import { useState } from 'react';
import { classNames } from '../utils/misc';

export default function ChatInputExtraContextItem({
  items,
  removeItem,
  clickToShow,
}: {
  items?: MessageExtra[];
  removeItem?: (index: number) => void;
  clickToShow?: boolean;
}) {
  const [show, setShow] = useState(-1);
  const showingItem = show >= 0 ? items?.[show] : undefined;

  if (!items) return null;

  return (
    <div
      className="flex flex-row gap-4 overflow-x-auto py-2 px-1 mb-1"
      role="group"
      aria-description="Selected files"
    >
      {items.map((item, i) => (
        <div
          className="indicator"
          key={i}
          onClick={() => clickToShow && setShow(i)}
          tabIndex={0}
          aria-description={
            clickToShow ? `Click to show: ${item.name}` : undefined
          }
          role={clickToShow ? 'button' : 'menuitem'}
        >
          {removeItem && (
            <div className="indicator-item indicator-top">
              <button
                aria-label="Remove file"
                className="btn btn-neutral btn-sm w-4 h-4 p-0 rounded-full"
                onClick={() => removeItem(i)}
              >
                <XMarkIcon className="h-3 w-3" />
              </button>
            </div>
          )}

          <div
            className={classNames({
              'flex flex-row rounded-md shadow-sm items-center m-0 p-0': true,
              'cursor-pointer hover:shadow-md': !!clickToShow,
            })}
          >
            {item.type === 'imageFile' ? (
              <>
                <img
                  src={item.base64Url}
                  alt={`Preview image for ${item.name}`}
                  className="w-14 h-14 object-cover rounded-md"
                />
              </>
            ) : (
              <>
                <div
                  className="w-14 h-14 flex items-center justify-center"
                  aria-description="Document icon"
                >
                  {item.type === 'audioFile' ? (
                    <SpeakerWaveIcon className="h-8 w-8 text-gray-500" />
                  ) : (
                    <DocumentTextIcon className="h-8 w-8 text-gray-500" />
                  )}
                </div>

                <div className="text-xs pr-4">
                  <b>{item.name ?? 'Extra content'}</b>
                </div>
              </>
            )}
          </div>
        </div>
      ))}

      {showingItem && (
        <dialog
          className="modal modal-open"
          aria-description={`Preview ${showingItem.name}`}
        >
          <div className="modal-box">
            <div className="flex justify-between items-center mb-4">
              <b>{showingItem.name ?? 'Extra content'}</b>
              <button
                className="btn btn-ghost btn-sm"
                aria-label="Close preview dialog"
              >
                <XMarkIcon className="h-5 w-5" onClick={() => setShow(-1)} />
              </button>
            </div>
            {showingItem.type === 'imageFile' ? (
              <img
                src={showingItem.base64Url}
                alt={`Preview image for ${showingItem.name}`}
              />
            ) : showingItem.type === 'audioFile' ? (
              <audio
                controls
                className="w-full"
                aria-description={`Audio file ${showingItem.name}`}
              >
                <source
                  src={`data:${showingItem.mimeType};base64,${showingItem.base64Data}`}
                  type={showingItem.mimeType}
                  aria-description={`Audio file ${showingItem.name}`}
                />
                Your browser does not support the audio element.
              </audio>
            ) : (
              <div className="overflow-x-auto">
                <pre className="whitespace-pre-wrap break-words text-sm">
                  {showingItem.content}
                </pre>
              </div>
            )}
          </div>
          <div className="modal-backdrop" onClick={() => setShow(-1)}></div>
        </dialog>
      )}
    </div>
  );
}
